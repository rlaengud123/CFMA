import os
from datetime import datetime

import neptune
import torch
import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader
import numpy as np

from torchfm.dataset.avazu import AvazuDataset
from torchfm.dataset.criteo import CriteoDataset
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.dataset.mydata import MyDataset
from torchfm.model.afi import AutomaticFeatureInteractionModel
from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.afn import AdaptiveFactorizationNetwork
from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.dfm import DeepFactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.hofm import HighOrderFactorizationMachineModel
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.cfm import ConvFM

def MAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def MBE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true-y_pred)

def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    elif name == 'mydata':
        return MyDataset(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset, embed_dim=256):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim)
    elif name == 'hofm':
        return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=embed_dim)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim, mlp_dims=(512, 512), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim, mlp_dims=(512, 512), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim, mlp_dims=(512,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim, mlp_dims=(512,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim, num_layers=3, mlp_dims=(512, 512), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim, mlp_dims=(512,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx, embed_dim=embed_dim, mlp_dims=(512, 512), dropout=0.2)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim, mlp_dims=(512,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim, mlp_dims=(512, 512), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(512, 512), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim, attn_size=256, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim, atten_embed_dim=256, num_heads=2, num_layers=3, mlp_dims=(512, 512), dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim, LNN_dim=2048, mlp_dims=(512, 512, 512), dropouts=(0.2, 0.2, 0.2))

    elif name == 'cfm':
        return ConvFM(field_dims, embed_dim, attn_size=256, dropouts=(0.2, 0.2))
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    # data_loader = train_data_loader
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device).long(), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, criterion, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device).long(), target.to(device)
            y = model(fields)
            loss = criterion(y, target.float())
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return mean_squared_error(targets, predicts), mean_absolute_error(targets, predicts), MAPE(targets, predicts), MBE(targets, predicts), loss


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         embed_dim):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.64)
    valid_length = int(len(dataset) * 0.18)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    model = get_model(model_name, dataset, embed_dim).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=10, save_path=f'{save_dir}/{model_name}_best.pt')
    folder = str(datetime.now())[:19].replace(':','-').replace(' ','_')

    save_dir = os.path.join(save_dir, folder) + f'_{model_name}'

    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, 'cp'))

    best_loss = 100
    best_mse = 100
    best_mae = 100
    best_mbe = 100
    best_mape = 100


    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        mse, mae, mape, mbe, loss = test(model, valid_data_loader, criterion, device)
        print('epoch:', epoch_i, 'validation: mse:', mse)
        neptune.log_metric('Val loss', loss)
        neptune.log_metric('Val mse', mse)
        neptune.log_metric('Val mae', mae)
        neptune.log_metric('Val mape', mape)
        neptune.log_metric('Val mbe', mbe)
        if not early_stopper.is_continuable(model, loss):
            print('This is the best')
            best_loss = loss
            best_mse = mse
            best_mae = mae
            best_mape = mape
            best_mbe = mbe
            neptune.log_metric('best loss', best_loss)
            neptune.log_metric('best mse', best_mse) 
            neptune.log_metric('best mae', best_mae) 
            neptune.log_metric('best mape', best_mape)
            neptune.log_metric('best mbe', best_mbe) 
            neptune.log_text('best epoch', str(epoch_i))


    mse, mae, mape, mbe, loss = test(model, test_data_loader, criterion, device)
    print(f'test mse: {mse}')
    neptune.log_metric('test mse', mse)
    neptune.log_metric('test mae', mae) 
    neptune.log_metric('test mape', mape)
    neptune.log_metric('test mbe', mbe) 
    neptune.log_metric('test loss', loss)
    neptune.stop()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='mydata')
    parser.add_argument('--dataset_path', default='data\mydata\mydata.csv',help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat')
    parser.add_argument('--model_name', default='afi')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--embed_dim', default=128, type=int)
    parser.add_argument('--tag', default='')
    parser.add_argument('--description', type=str)
    args = parser.parse_args()

    neptune.init(project_qualified_name='ninyo/recom-ieee',api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNTE4Yjg0MzEtMjYyYS00NzVlLTg4MjAtZGNiZGJhYThkY2Q4In0=')

    tag_list = args.tag.split(',')

    def NeptuneLog():
        neptune.log_text('model',args.model_name) 
        neptune.log_text('save_dir',args.save_dir)
        if tag_list != ['']:
            neptune.append_tag(tag_list)


    neptune.create_experiment(args.model_name, params=vars(args), description=args.description)
    NeptuneLog()

    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.embed_dim)



# args = {"dataset_name" : "mydata",
#         "dataset_path" : "data\mydata\mydata.csv",
#         "model_name" : "afm",
#         "epoch" : 100,
#         "learning_rate" : 0.001,
#         "batch_size" : 128,
#         "weight_decay" : 1e-6,
#         "device" : "cuda:0",
#         "save_dir" : "chkpt",
#         "embed_dim" : 512,
        
#         "nhead" : 8,
#         "num_ecoder_layers" : 6,
#         "num_dencoder_layers" : 6,
#         "dim_feedforward" : 2048,
#         "dropout" : 0.1,
#         "activation" : "relu",
#         }

# dataset_name,dataset_path,model_name,epoch,learning_rate,batch_size,weight_decay,device,save_dir,embed_dim = args["dataset_name"],args["dataset_path"],args["model_name"],args["epoch"],args["learning_rate"],args["batch_size"],args["weight_decay"],args["device"],args["save_dir"],args["embed_dim"]

# nhead, num_ecoder_layers, num_dencoder_layers, dim_feedforward, dropout, activation = args["nhead"],args["num_ecoder_layers"],args["num_dencoder_layers"],args["dim_feedforward"],args["dropout"],args["activation"]