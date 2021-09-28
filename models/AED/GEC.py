import torch
import os
import numpy as np
import pandas as pd
import dgl
import dgl.data
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

import torch.nn as nn
import torch.nn.functional as F

from make_dgl_dataset import PmuDataset
import warnings
from dgl.nn import GraphConv
from dgl.nn import GATConv
import matplotlib.pyplot as plt

# from sequitur.models import LSTM_AE
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from models.AED.simpleAED import Encoder, Decoder, RecurrentAutoencoder, LatentMu
import copy
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from torch import nn, optim
# import pandas as pd
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch.autograd import Variable


class GEC(nn.Module):
    def __init__(self, in_feats, h1_feats, last_space_feature):
        super(GEC, self).__init__()
        self.conv1 = GATConv(in_feats, h1_feats, 1)
        # self.conv2 = GraphConv(h1_feats, h2_feats)
        self.conv2 = GATConv(h1_feats, last_space_feature, 1)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        # h = self.conv3(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

# Create the model with given dimensions


