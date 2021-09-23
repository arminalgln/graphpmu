import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
import numpy as np
import pandas as pd
from make_dgl_dataset import PmuDataset
import warnings
from dgl.nn import GraphConv
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)

features = np.load('data/features.npy')
# labels = np.load('data/labels.npy')
data = np.load('data/all_event_data.npy')
# per_unit = np.load('data/all_per_unit.npy')
per_unit = np.load('data/aug_all_per_unit_806_824_836_846.npy')
labels = np.load('data/aug_labels_806_824_836_846.npy')


dataset = PmuDataset()

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

# num_examples = len(dataset)
# num_train = int(num_examples * 0.9)
#
# train_sampler = SubsetRandomSampler(torch.arange(num_train))
# test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_selector = np.random.choice(num_examples, num_train, replace=False)
test_selector = np.setdiff1d(np.arange(num_examples), train_selector)


train_sampler = SubsetRandomSampler(torch.from_numpy(train_selector))
test_sampler = SubsetRandomSampler(torch.from_numpy(test_selector))

#%%
b_size = 100

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=b_size, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=b_size, drop_last=False)


class GCN(nn.Module):
    def __init__(self, in_feats, h1_feats, h2_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h1_feats)
        self.conv2 = GraphConv(h1_feats, h2_feats)
        self.conv3 = GraphConv(h2_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.leaky_relu(h)
        h = self.conv2(g, h)
        h = F.leaky_relu(h)
        h = self.conv3(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')

# Create the model with given dimensions

#%%
model = GCN(9*125, 512, 256, np.unique(labels).shape[0])
# model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epoch_number = 20
for epoch in range(epoch_number):
    acc = 0
    rnd = 0
    for batched_graph, tags in train_dataloader:
        num_correct = 0
        num_trains = 0
        pred = model(batched_graph, batched_graph.ndata['features'].float())
        loss = F.cross_entropy(pred, tags)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_correct += (pred.argmax(1) == tags).sum().item()
        # print(pred.argmax(1), labels)
        num_trains += len(tags)
        acc += num_correct / num_trains
        rnd += 1
    print(epoch, acc/rnd)
        # print('Test accuracy:', num_correct / num_trains)
        # print(pred.argmax(1), labels)
    # print(epoch)

from sklearn import metrics
num_correct = 0
num_tests = 0
y_pred = np.array([])
y_real = np.array([])
for batched_graph, tags in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['features'].float())
    num_correct += (pred.argmax(1) == tags).sum().item()
    # print(pred.argmax(1), tags)
    pred = pred.detach().numpy()
    tags = tags.detach().numpy()
    # print(pred.argmax(1))
    num_tests += len(tags)
    y_pred = np.append(y_pred, pred.argmax(1).ravel())
    y_real = np.append(y_real, tags.ravel())

# y_pred = np.ravel(y_pred)
# y_real = np.ravel(y_real)

print('test accuracy (ARS)', metrics.adjusted_rand_score(y_real, y_pred))


for batched_graph, tags in train_dataloader:
    pred = model(batched_graph, batched_graph.ndata['features'].float())
    num_correct += (pred.argmax(1) == tags).sum().item()
    # print(pred.argmax(1), tags)
    pred = pred.detach().numpy()
    tags = tags.detach().numpy()
    # print(pred.argmax(1))
    num_tests += len(tags)
    y_pred = np.append(y_pred, pred.argmax(1).ravel())
    y_real = np.append(y_real, tags.ravel())

# y_pred = np.ravel(y_pred)
# y_real = np.ravel(y_real)
print('trian accuracy (ARS)', metrics.adjusted_rand_score(y_real, y_pred))

#%%
num_correct = 0
num_tests = 0
for batched_graph, tags in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['features'].float())
    num_correct += (pred.argmax(1) == tags).sum().item()
    # print(pred.argmax(1), tags)
    num_tests += len(tags)

print('Test accuracy:', num_correct / num_tests)
#%%
from sklearn import metrics
num_correct = 0
num_tests = 0
y_pred = np.array([])
y_real = np.array([])
for batched_graph, tags in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['features'].float())
    num_correct += (pred.argmax(1) == tags).sum().item()
    # print(pred.argmax(1), tags)
    pred = pred.detach().numpy()
    tags = tags.detach().numpy()
    print(pred.argmax(1))
    num_tests += len(tags)
    y_pred = np.append(y_pred, pred.argmax(1).ravel())
    y_real = np.append(y_real, tags.ravel())

# y_pred = np.ravel(y_pred)
# y_real = np.ravel(y_real)

print(metrics.adjusted_rand_score(y_real, y_pred))


print('Test accuracy:', num_correct / num_tests)

#%%
import networkx as nx
import dgl

g_nx = nx.petersen_graph()
g_dgl = dgl.DGLGraph(g_nx)

import matplotlib.pyplot as plt
# plt.subplot(121)
# nx.draw(g_nx, with_labels=True)
plt.subplot(111)
nx.draw(dataset[0][0].to_networkx(), with_labels=True)

plt.show()
