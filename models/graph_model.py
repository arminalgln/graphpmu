import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from models.discriminator import Discriminator
from models.losses import global_loss_


class GraphEncoder(nn.Module):
    def __init__(self, node_nums, in_feats, h1_feats, last_space_feature):
        super(GraphEncoder, self).__init__()
        # self.conv1 = GATConv(in_feats, h1_feats, 1)
        self.node_nums = node_nums
        self.in_feats = in_feats
        self.last_space_feature = last_space_feature
        self.conv1 = GraphConv(in_feats, h1_feats)
        self.conv2 = GraphConv(h1_feats, last_space_feature)
        # self.conv2 = GATConv(h1_feats, last_space_feature, 1)
        # self.device = device
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, g):
        batch_size = int(g.num_nodes() / self.node_nums)
        h = self.conv1(g, g.ndata['features'])
        h = F.leaky_relu(h)
        h1 = h
        g.ndata['h1'] = h1
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h2 = h
        # H = torch.reshape(h2, (batch_size, self.node_nums, self.last_space_feature))
        # H = torch.mean(H, axis=1)
        g.ndata['h2'] = h2
        H = dgl.readout.mean_nodes(g, 'h2')
        # g.ndata['H'] = H
        return H

class GraphPMU(nn.Module):
    def __init__(self, encoder, discriminator, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2, device):
        super(GraphPMU, self).__init__()
        self.node_nums = node_nums
        self.in_feats = in_feats
        self.h1_feats = h1_feats
        self.last_space_feature = last_space_feature
        self.D_h1 = D_h1
        self.D_h2 = D_h2
        # self.measure = measure
        self.device = device
        self.encoder = encoder(node_nums, in_feats, h1_feats, last_space_feature).to(device)
        # self.discriminator = discriminator(last_space_feature, D_h1, D_h2).to(device)#if simple encoder
        self.discriminator = discriminator(last_space_feature + h1_feats, D_h1, D_h2).to(device)#if locglob encoder
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, g):
        H = self.encoder(g.to(self.device))
        y = self.discriminator(H)

        return y



class GraphEncoderLocGlob(nn.Module):
    def __init__(self, node_nums, in_feats, h1_feats, last_space_feature):
        super(GraphEncoderLocGlob, self).__init__()
        # self.conv1 = GATConv(in_feats, h1_feats, 1)
        self.node_nums = node_nums
        self.in_feats = in_feats
        self.last_space_feature = last_space_feature
        self.conv1 = GraphConv(in_feats, h1_feats)
        self.conv2 = GraphConv(h1_feats, last_space_feature)
        # self.conv2 = GATConv(h1_feats, last_space_feature, 1)
        # self.device = device
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, g):
        batch_size = int(g.num_nodes() / self.node_nums)
        h = self.conv1(g, g.ndata['features'])
        h = F.leaky_relu(h)
        h1 = h
        g.ndata['h1'] = h1
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h2 = h
        # H = torch.reshape(h2, (batch_size, self.node_nums, self.last_space_feature))
        # H = torch.mean(H, axis=1)
        g.ndata['h2'] = h2
        g.ndata['hcat'] = torch.cat((h1, h2), 1)
        H = dgl.readout.mean_nodes(g, 'hcat')
        # g.ndata['H'] = H
        return H

class GraphPMULocalGlobal(nn.Module):
    def __init__(self, encoder, discriminator, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2, device):
        super(GraphPMULocalGlobal, self).__init__()
        self.node_nums = node_nums
        self.in_feats = in_feats
        self.h1_feats = h1_feats
        self.last_space_feature = last_space_feature
        self.D_h1 = D_h1
        self.D_h2 = D_h2
        # self.measure = measure
        self.device = device
        self.encoder = encoder(node_nums, in_feats, h1_feats, last_space_feature).to(device)
        # self.discriminator = discriminator(last_space_feature, D_h1, D_h2).to(device)#if simple encoder
        self.discriminator = discriminator(2 * (last_space_feature + h1_feats), D_h1, D_h2).to(device)#if locglob encoder
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, g):
        H = self.encoder(g.to(self.device)) # encoder should be GraphEncoderLocGlob
        y = self.discriminator(H)

        return y
