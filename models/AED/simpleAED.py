import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from dgl.nn import GATConv

import torch.autograd.function as Function


class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim):
    super(Encoder, self).__init__()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = device
    self.seq_len, self.n_features = seq_len, n_features
    self.rnn1_dim, self.rnn2_dim = embedding_dim, 2 * embedding_dim
    self.dense1_dim, self.embedding_dim = 2 * embedding_dim, embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=self.n_features,
      hidden_size=self.rnn1_dim,
      num_layers=1,
      #dropout=0.4,
      batch_first=True  # (batch, seq, feature) as output and (num_layers, batch, hidden_size) for h, c
    )  # (batch, seq, feature)  for input if batch_first is true
    self.rnn2 = nn.LSTM(
      input_size=self.rnn1_dim,
      hidden_size=self.dense1_dim,
      num_layers=1,
      #dropout=0.4,
      batch_first=True
    )
    self.dense1 = nn.Linear(
        in_features=self.dense1_dim,
        out_features=self.embedding_dim
    )
    # self.batchnorm = nn.BatchNorm1d(self.embedding_dim)

  def forward(self, x):
    # x.to(self.device)
    x = x.float()
    x, (_, _) = self.rnn1(x)
    x = F.leaky_relu(x)
    _, (hidden_n, _) = self.rnn2(x)
    h_size = hidden_n.shape
    x = hidden_n.reshape(h_size[1], h_size[2])
    x = F.leaky_relu(x)
    x = self.dense1(x)
    # x = self.batchnorm(x) #some models have this some not

    return torch.tanh(x)


class Decoder(nn.Module):
  def __init__(self, seq_len, n_features, input_dim):
      super(Decoder, self).__init__()
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.device = device
      self.seq_len, self.input_dim, self.n_features = seq_len, input_dim, n_features
      self.dense1_dim, self.rnn1_dim, self.rnn2_dim = self.seq_len * self.input_dim * 2, 2 * input_dim, input_dim

      self.dense1 = nn.Linear(
          in_features=self.input_dim,
          out_features=self.dense1_dim
      )
      self.rnn1 = nn.LSTM(
          input_size=self.rnn1_dim,
          hidden_size=self.rnn2_dim,
          num_layers=1,
          batch_first=True
      )
      self.rnn2 = nn.LSTM(
          input_size=self.rnn2_dim,
          hidden_size=self.n_features,
          num_layers=1,
          batch_first=True
      )

  def forward(self, x):
      # x.to(self.device)
      x = x.float()
      x = self.dense1(x)
      x = F.leaky_relu(x)
      size = x.shape
      x = x.reshape((size[0], self.seq_len, self.rnn1_dim))
      x, (_, _) = self.rnn1(x)
      x = F.leaky_relu(x)
      x, (_, _) = self.rnn2(x)
      return F.leaky_relu(x)


class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = device
    super(RecurrentAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.encoder.float()
    self.decoder = Decoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder.float()

  def forward(self, x):
    # x = x.to(self.device)
    x = x.float()
    x = self.encoder(x)
    x = self.decoder(x)
    return x

class GEC(nn.Module):
    def __init__(self, in_feats, h1_feats, last_space_feature):
        super(GEC, self).__init__()
        # self.conv1 = GATConv(in_feats, h1_feats, 1)
        self.conv1 = GraphConv(in_feats, h1_feats)
        self.conv2 = GraphConv(h1_feats, last_space_feature)
        # self.conv2 = GATConv(h1_feats, last_space_feature, 1)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        # h = self.conv3(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

class EncGraph(nn.Module):
  def __init__(self, encoder, graph, pmu_number, src, dst):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = device
    super(EncGraph, self).__init__()

    self.encoder = encoder
    self.graph = graph.to(self.device)
    self.pmu_number = pmu_number
    self.src = src
    self.dst = dst

  def forward(self, x):
    # x = x.to(self.device)
    x = x.float()
    y = np.split(x, self.pmu_number, axis=-1)
    z = [self.encoder(i) for i in y] #all latent z for each pmu in a batch
    size = z[0].shape
    #TODO
    #define the graph to pass it to the self.graph
    # z = torch.cat([x.float() for x in z], dim=0).reshape(self.pmu_number, size[0], size[1]).to(self.device)
    z = torch.cat([x.float() for x in z], dim=0).reshape(self.pmu_number, size[0], size[1])
    # print(z.shape)
    x = []
    for idx in range(size[0]):
        g = dgl.graph((self.src, self.dst), num_nodes=self.pmu_number)
        g = dgl.add_self_loop(g)
        g = g.to(self.device)
        g.ndata['features'] = z[:, idx].clone()
        x.append(g)
    # print(len(x))
    x = dgl.batch(x)
    # print(x)

    x = self.graph(x, x.ndata['features'])
    xs = x.shape
    x = x.reshape(xs[0], xs[-1]).clone()

    return x


class LatentMu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mu):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ctx.device = device
        z = z.float()
        mu = mu.float()
        ctx.alpha = 1
        ctx.mu = mu
        ctx.save_for_backward(z, mu)

        def q_measure(ctx, z, mu):
            p = 1.0
            dist = torch.cdist(z, mu, 2).float()
            q = 1.0 / (1 + dist ** 2 / ctx.alpha)
            q = (q.T / q.sum(axis=1)).T
            # print('q: ',q[0])
            return q

        def aux_dist(ctx, z, mu):
            q = q_measure(ctx, z, mu)
            q = (q.T / q.sum(axis=1)).T
            p = (q ** 2)
            p = (p.T / p.sum(axis=1)).T

            return p

        def loss_fun(ctx, z):
            # q = q_measure(ctx, z, mu).to(ctx.device)
            q = q_measure(ctx, z, mu)
            # p = aux_dist(ctx, z, mu).to(ctx.device)
            p = aux_dist(ctx, z, mu)
            return F.kl_div(p, q, reduction='batchmean')

        # print(loss_fun(ctx, z))
        return loss_fun(ctx, z)

    @staticmethod
    def backward(ctx, grad_output):
        z, mu = ctx.saved_tensors
        def q_measure(ctx, z, mu):
            p = 1.0
            dist = torch.cdist(z, mu, 2).float()
            q = 1.0 / (1 + dist ** 2 / ctx.alpha)#TODO think about other measures as I removed 1+
            q = (q.T / q.sum(axis=1)).T
            # print('q: ', q[0])
            return q

        def aux_dist(ctx, z, mu):
            z = z
            q = q_measure(ctx, z, mu)
            q = (q.T / q.sum(axis=1)).T
            p = (q ** 2)
            p = (p.T / p.sum(axis=1)).T
            # print('p: ', p[0])
            return p

        def grad(ctx, z, mu):
            # z = z.float().to(ctx.device)
            z = z.float()
            # mu = mu.float().to(ctx.device)
            mu = mu.float()
            q = q_measure(ctx, z, mu)
            p = aux_dist(ctx, z, mu)
            d = 1 + torch.cdist(z, mu, 2) ** 2
            constant_val = (p - q) / d

            constant_val_T = constant_val.T

            extented_z = z.repeat((mu.shape[0], 1))
            extented_mu = torch.repeat_interleave(mu, z.shape[0], dim=0)
            simple_dist_z_mu = (extented_z - extented_mu).reshape(mu.shape[0], z.shape[0], z.shape[-1])

            constant_val_T_rep = constant_val_T.repeat_interleave(z.shape[-1], axis=1)
            constant_val_T_rep = constant_val_T_rep.reshape(mu.shape[0], z.shape[0], z.shape[-1])

            all_grads = 2 * constant_val_T_rep * simple_dist_z_mu

            # z = z
            # q = q_measure(ctx, z, mu)
            # p = aux_dist(ctx, z, mu)
            # extented_z = z.repeat((mu.shape[0],1))
            # extented_mu = torch.repeat_interleave(mu, z.shape[0], dim = 0)
            # grad = 2 * (p - q) * torch.cdist(z, mu, 1) / (1 + torch.cdist(z, mu, 2)**2)
            # grad_z = torch.sum(grad, axis=1)
            # grad_mu = -torch.sum(grad, axis=0)

            return all_grads
        all_grads = grad(ctx, z, mu)
        # grad_z = torch.sum(all_grads, axis=0).to(ctx.device)
        grad_z = torch.sum(all_grads, axis=0)
        # grad_mu = -torch.sum(all_grads, axis=1).to(ctx.device)
        grad_mu = -torch.sum(all_grads, axis=1)
        # print(grad_z)
        # grad_z = grad_z.view(-1, 1).expand_as(z).to(ctx.device)
        # grad_mu = grad_mu.view(-1, 1).expand_as(mu).to(ctx.device)
        # print(grad_z)
        # grad_z = z.mul(grad_z) * grad_output
        # grad_mu = mu.mul(grad_mu) * grad_output
        # print(grad_output.shape)
        # print(grad_z.shape)
        # print(grad_mu.shape)
        return grad_z*grad_output, grad_mu*grad_output

# #%%
# #test latent
# z=torch.rand((3,2), requires_grad=True, device=device)
# m=torch.rand((2,2), requires_grad=True, device=device)
# ls=LatentMu.apply(z,m)
# ls.backward()


