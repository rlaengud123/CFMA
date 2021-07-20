import torch
import numpy as np
import torch.nn.functional as F

fields = torch.tensor([11,1])

field_dims = 500
embed_dim = 16
att_dim = 16

fields = fields+fields.new_tensor(np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long))

x = torch.nn.Embedding(field_dims, embed_dim)(fields)

num_fields = x.shape[1]

row, col = list(), list()

for i in range(num_fields - 1):
    for j in range(i + 1, num_fields):
        row.append(i), col.append(j)
p, q = x[:, row], x[:, col]
inner_product = p * q

attn_scores = F.relu(torch.nn.Linear(16,16)(inner_product))