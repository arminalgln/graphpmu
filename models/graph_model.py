import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from dgl.nn import GATConv

import torch.autograd.function as Function


class GraphEncoder(nn.Module):
  def __init__(self, in_feats, h1_feats, last_space_feature):
    super(GraphEncoder, self).__init__()
    # self.conv1 = GATConv(in_feats, h1_feats, 1)
    self.conv1 = GraphConv(in_feats, h1_feats)
    self.conv2 = GraphConv(h1_feats, last_space_feature)
    # self.conv2 = GATConv(h1_feats, last_space_feature, 1)

  def forward(self, g, in_feat):

    h = self.conv1(g, in_feat)
    h = F.leaky_relu(h)
    h1 = h
    h = self.conv2(g, h)
    h = F.leaky_relu(h)
    h2 = h

    return h1, h2, torch.mean(h2,axis=0)

#%%
import pickle
with open('data/positive_graphs.pkl', 'rb') as handle:
  pos_graphs = pickle.load(handle)
#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graph = dgl.batch(pos_graphs[1:10])
model = GraphEncoder(32, 64, 8).to(device)
h1, h2, H = model(graph, graph.ndata['features'])
import matplotlib.pyplot as plt
a=h1.detach().cpu().numpy()
b=h2.detach().cpu().numpy()
plt.imshow(a)
plt.show()
plt.imshow(b)
plt.show()
