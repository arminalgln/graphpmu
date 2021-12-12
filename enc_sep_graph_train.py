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

import warnings
warnings.filterwarnings("ignore")

# from sequitur.models import LSTM_AE
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from models.AED.simpleAED import Encoder, Decoder, RecurrentAutoencoder, LatentMu, EncGraph, GEC
import copy
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from torch import nn, optim
# import pandas as pd
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch.autograd import Variable
# from models.AED.GEC import GEC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#load the data
per_unit = np.load('data/aug_all_per_unit_806_824_836_846.npy')
labels = np.load('data/aug_labels_806_824_836_846.npy')

cluster_number = np.unique(labels).shape[0]
# normalize data
new_data = []

for i in range(per_unit.shape[-1]):
  mx = np.max(per_unit[:, :, i], axis=1)
  # mn = np.min(per_unit[:, :, i], axis=1)

  new_data.append((per_unit[:, :, i])/(mx[:, None]))

new_data = np.array(new_data)
new_data = np.swapaxes(new_data, 0, 1)
per_unit = np.swapaxes(new_data, 2, 1)


feature_number = 9  # include v, i and theta for all three phases
pmu_number = int(per_unit.shape[-1] / feature_number)

# separate the pmu data
new_labels = np.repeat(labels, pmu_number, axis=0)
for indx, data in enumerate(np.split(per_unit, pmu_number, axis=-1)):
    if indx == 0:
        new_per_unit = data
    else:
        new_per_unit = np.concatenate((new_per_unit, data), axis=0)
#
# print(per_unit.shape, new_per_unit.shape)
# print(labels.shape, new_labels.shape)

num_examples, seq_len, n_features = per_unit.shape

num_train = int(num_examples * 0.9)
train_selector = np.random.choice(num_examples, num_train, replace=False)
test_selector = np.setdiff1d(np.arange(num_examples), train_selector)

train_index = np.random.choice(train_selector.shape[0], 10000, replace=False)
test_index = np.random.choice(test_selector.shape[0], 1000, replace=False)

train_selector = train_selector[train_index]
test_selector = test_selector[test_index]


train_sampler = SubsetRandomSampler(torch.from_numpy(train_selector))
test_sampler = SubsetRandomSampler(torch.from_numpy(test_selector))

b_size = 100

train_dataloader = DataLoader(
    per_unit, sampler=train_sampler, batch_size=b_size, drop_last=False)
test_dataloader = DataLoader(
    per_unit, sampler=test_sampler, batch_size=b_size, drop_last=False)

new_train_dataloader = DataLoader(
    new_per_unit, sampler=train_sampler, batch_size=b_size, drop_last=False)
new_test_dataloader = DataLoader(
    new_per_unit, sampler=test_sampler, batch_size=b_size, drop_last=False)

zspace_feature_number = 16
#define and train a model and save
model = RecurrentAutoencoder(seq_len, feature_number, zspace_feature_number)
model.to(device)
model.float()

last_space_graph_feature = 16

graph_model = GEC(zspace_feature_number, 256, last_space_graph_feature)
graph_model.to(device)
graph_model.float()

# src = torch.from_numpy(np.array([0, 0, 0]))
# dst = torch.from_numpy(np.array([1, 2, 3]))
#%%
def train_AED(model, train_dataset, val_dataset, pmu_number, n_epochs):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for bat_num, batch_input in enumerate(train_dataset):
            splitted_batch = np.split(batch_input.to(device), pmu_number, axis=-1)

            for count, pmu_event_data in enumerate(splitted_batch):
                pmu_event_data = pmu_event_data
                # train AED with reconstruction loss
                predicted = model(pmu_event_data)
                optimizer.zero_grad()
                loss = criterion(predicted.float(), pmu_event_data.float())
                loss.backward()
                train_losses.append(loss.item())
                optimizer.step()

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for bat_num, batch_input in enumerate(val_dataset):
                splitted_batch = np.split(batch_input.to(device), pmu_number, axis=-1)

                for count, pmu_event_data in enumerate(splitted_batch):
                    pmu_event_data = pmu_event_data
                    pred = model(pmu_event_data)
                    loss = criterion(pred.float(), pmu_event_data.float())
                    val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            model_path = 'models/AED/806_824_836_846_just_AED'
            torch.save(model, model_path)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    return model.eval()


model = train_AED(
    model, train_dataloader, test_dataloader,
    pmu_number=pmu_number, n_epochs=20
)
#%%
model_lat = RecurrentAutoencoder(seq_len, feature_number, zspace_feature_number)
model_lat.to(device)
model_lat.float()
def update_mu(final_latent, device):
    final_latent = final_latent.detach().cpu()
    kmeans = KMeans(9, n_init=20)
    kmeans.fit(final_latent)
    # mu = torch.from_numpy(kmeans.cluster_centers_).to(device)
    mu = torch.tensor(kmeans.cluster_centers_, requires_grad=True, device=device)
    # mu = Variable(torch.from_numpy(kmeans.cluster_centers_), requires_grad=True).to(device)
    return mu
def train_AED_latentmu(model, train_dataset, val_dataset, pmu_number, n_epochs):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    enc = model.encoder
    enc = enc.train()
    initial = True
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        lmu_loss = []
        for bat_num, batch_input in enumerate(train_dataset):
            splitted_batch = np.split(batch_input.to(device), pmu_number, axis=-1)


            for count, pmu_event_data in enumerate(splitted_batch):
                pmu_event_data = pmu_event_data
                # train AED with reconstruction loss
                predicted = model(pmu_event_data)
                optimizer.zero_grad()
                loss = criterion(predicted.float(), pmu_event_data.float())
                if count == 0:
                    z = enc(pmu_event_data)
                else:
                    z = torch.cat((z,enc(pmu_event_data)))
                train_losses.append(loss.item())
                loss.backward()

            if initial:
                mu = update_mu(z, device)
                initial = False
            loss = LatentMu.apply(z, mu)
            lmu_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        mu.data -= mu.grad.data * 0.001
        mu.grad.data.zero_()

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for bat_num, batch_input in enumerate(val_dataset):
                splitted_batch = np.split(batch_input.to(device), pmu_number, axis=-1)

                for count, pmu_event_data in enumerate(splitted_batch):
                    pmu_event_data = pmu_event_data
                    pred = model(pmu_event_data)
                    loss = criterion(pred.float(), pmu_event_data.float())
                    val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lmu_loss = np.mean(lmu_loss)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            model_path = 'models/AED/806_824_836_846_just_AED'
            torch.save(model, model_path)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss} lmuloss {lmu_loss}' )
    model.load_state_dict(best_model_wts)
    return model.eval()


model_lat = train_AED_latentmu(
    model_lat, train_dataloader, test_dataloader,
    pmu_number=pmu_number, n_epochs=300
)

#%%
src = torch.from_numpy(np.array([0, 1, 2]))
dst = torch.from_numpy(np.array([1, 2, 3]))
def batchdata_to_latentgraph(model, batch_input, pmu_number, src, dst):
    model = model.eval()
    enc = model.encoder
    enc = enc.eval()
    with torch.no_grad():
        # for bat_num, batch_input in enumerate(train_dataset):
        splitted_batch = np.split(batch_input.to(device), pmu_number, axis=-1)

        batch_graph = []

        z = [enc(i) for i in splitted_batch]  # all latent z for each pmu in a batch
        size = z[0].shape
        z = torch.cat([x.float() for x in z], dim=0).reshape(pmu_number, size[0], size[1])
        for idx in range(size[0]):
            g = dgl.graph((src, dst), num_nodes=pmu_number)
            g = dgl.add_self_loop(g)
            g = g.to(device)
            g.ndata['features'] = z[:, idx].clone().detach()
            batch_graph.append(g)
        bg = dgl.batch(batch_graph)
    return bg


def update_mu(final_latent, device):
    final_latent = final_latent.detach().cpu()
    kmeans = KMeans(9, n_init=20)
    kmeans.fit(final_latent)
    # mu = torch.from_numpy(kmeans.cluster_centers_).to(device)
    mu = torch.tensor(kmeans.cluster_centers_, requires_grad=True, device=device)
    # mu = Variable(torch.from_numpy(kmeans.cluster_centers_), requires_grad=True).to(device)
    return mu
def train_graph(model, graph_model, train_dataset, val_dataset, pmu_number, n_epochs, src, dst, device):
    graph_model = graph_model.train()
    graph_optimizer = torch.optim.Adam(graph_model.parameters(), lr=0.001)
    initial = True
    best_loss=10000
    for epoch in range(1, n_epochs + 1):
        model = model.train()
        train_losses = []
        for bat_num, batch_input in enumerate(train_dataset):
            bg = batchdata_to_latentgraph(model, batch_input, pmu_number, src, dst)
            graph_optimizer.zero_grad()
            latents = graph_model(bg, bg.ndata['features'])
            if initial:
                mu = update_mu(latents, device)
                initial = False
            loss = LatentMu.apply(latents, mu)
            loss.backward()
            train_losses.append(loss.item())
            graph_optimizer.step()
        mu.data -= 0.001 * mu.grad.data
        mu.grad.data.zero_()


        val_losses = []
        graph_model = graph_model.eval()
        with torch.no_grad():
            for bat_num, batch_input in enumerate(val_dataset):
                bg = batchdata_to_latentgraph(model, batch_input, pmu_number, src, dst)
                latents = graph_model(bg, bg.ndata['features'])
                loss = LatentMu.apply(latents, mu)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        if np.abs(val_loss) < np.abs(best_loss):
            best_loss = val_loss
            best_model_wts = copy.deepcopy(graph_model.state_dict())
            model_path = 'models/AED/806_824_836_846_graph_after_AED'
            torch.save(graph_model, model_path)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    graph_model.load_state_dict(best_model_wts)

    return graph_model.eval()

graph_model = train_graph(model, graph_model, train_dataloader, test_dataloader, pmu_number, 20, src, dst, device)


#%%
#
# def update_mu(final_latent, device):
#     final_latent = final_latent.detach().cpu()
#     kmeans = KMeans(9, n_init=20)
#     kmeans.fit(final_latent)
#     # mu = torch.from_numpy(kmeans.cluster_centers_).to(device)
#     mu = torch.tensor(kmeans.cluster_centers_, requires_grad=True, device=device)
#     # mu = Variable(torch.from_numpy(kmeans.cluster_centers_), requires_grad=True).to(device)
#     return mu
# def train_model(model, graph_model, train_dataset, val_dataset, pmu_number, n_epochs):
#     enc = model.encoder
#     dec = model.decoder
#
#     criterion = nn.MSELoss(reduction='mean')
#     enc_criteria = LatentMu.apply
#     mu_learning_rate = 1e-3
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     enc_optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-4)
#     dec_optimizer = torch.optim.Adam(dec.parameters(), lr=1e-3, weight_decay=1e-4)
#     graph_optimizer = torch.optim.Adam(graph_model.parameters(), lr=1e-3)
#
#     src = torch.from_numpy(np.array([0, 1, 2]))
#     dst = torch.from_numpy(np.array([1, 2, 3]))
#     enocdergraph = EncGraph(model.encoder, graph_model, pmu_number, src, dst)
#
#     endGEC_optimizer = torch.optim.Adam(enocdergraph.parameters(), lr=1e-3, weight_decay=1e-4)
#
#     dec_criterion = nn.MSELoss(reduction='mean')
#     history = dict(train=[], val=[])
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_loss = 10000.0
#     #initial mu for 1000 events latent
#     # mu = update_mu(train_dataloader, enc, device)
#     initial = True
#     for epoch in range(1, n_epochs + 1):
#         model = model.train()
#         enc = model.encoder
#         enc = enc.train()
#         dec = model.decoder
#
#         train_losses = []
#         for bat_num, batch_input in enumerate(train_dataset):
#             splitted_batch = np.split(batch_input, pmu_number, axis=-1)
#             # print('bug')
#
#             for count, pmu_event_data in enumerate(splitted_batch):
#                 pmu_event_data = pmu_event_data.to(device)
#                 # train AED with reconstruction loss
#                 predicted = model(pmu_event_data)
#                 optimizer.zero_grad()
#                 loss = criterion(predicted.float(), pmu_event_data.float())
#                 loss.backward()
#                 train_losses.append(loss.item())
#                 optimizer.step()
#                 train_losses.append(loss.item())
#         # print(np.mean(train_losses))
#         val_losses = []
#         val_losses_enc = []
#         model = model.eval()
#         enc = enc.eval()
#         with torch.no_grad():
#             for bat_num, batch_input in enumerate(val_dataset):
#                 splitted_batch = np.split(batch_input, pmu_number, axis=-1)
#
#                 for count, pmu_event_data in enumerate(splitted_batch):
#                     pmu_event_data = pmu_event_data.to(device)
#                     pred = model(pmu_event_data)
#                     loss = criterion(pred.float(), pmu_event_data.float())
#                     val_losses.append(loss.item())
#                     # print(loss.item())
#
#         train_loss = np.mean(train_losses)
#         val_loss = np.mean(val_losses)
#         history['train'].append(train_loss)
#         history['val'].append(val_loss)
#         if val_loss < best_loss:
#             best_loss = val_loss
#             best_model_wts = copy.deepcopy(model.state_dict())
#             # model_path = 'models/AED/806_824_836_846_graph_AED'
#             # torch.save(model, model_path)
#             # model_path = 'models/AED/806_824_836_846_graph_graph'
#             # torch.save(graph_model, model_path)
#         # print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}  val loss enc {val_loss_enc}')
#         print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
#     for epoch in range(1, n_epochs + 1):
#         model = model.eval()
#         enc = model.encoder
#         enc = enc.eval()
#
#         graph_model = graph_model.train()
#
#         train_losses = []
#         all_final = []
#         for bat_num, batch_input in enumerate(train_dataset):
#
#             splitted_batch = np.split(batch_input.to(device), pmu_number, axis=-1)
#             batch_graph = []
#             # for count, pmu_event_data in enumerate(splitted_batch):
#             # splitted_batch = splitted_batch
#
#             # y = np.split(splitted_batch, pmu_number, axis=-1)
#             z = [enc(i) for i in splitted_batch]  # all latent z for each pmu in a batch
#             size = z[0].shape
#             z = torch.cat([x.float() for x in z], dim=0).reshape(pmu_number, size[0], size[1])
#             for idx in range(size[0]):
#                 g = dgl.graph((src, dst), num_nodes=pmu_number)
#                 g = dgl.add_self_loop(g)
#                 g = g.to(device)
#                 g.ndata['features'] = z[:, idx].clone().detach()
#                 # x.append(g)
#             # print(len(x))
#                 batch_graph.append(g)
#             # print(x)
#             bg = dgl.batch(batch_graph)
#             latents = graph_model(bg, bg.ndata['features'])
#             graph_optimizer.zero_grad()
#             # xs = latents.shape
#             # latents = latents.reshape(xs[0], xs[-1]).clone()
#             # latents.shape
#             # train AED with reconstruction loss
#             if initial:
#                 mu = update_mu(latents, device)
#                 initial = False
#
#             graph_loss = LatentMu.apply(latents, mu)
#             graph_loss.backward()
#             # print('mu_before', mu.grad[0, 0])
#             print('-----------------------------')
#             # endGEC_optimizer.zero_grad()
#             # graph_loss.backward()
#             # print('mu_before', mu[0, 0])
#             mu.data -= 1e-3 * mu.grad.data
#             # print('mu_before',mu[0,0])
#             mu.grad.data.zero_()
#             # print('mu_a', mu[0, 0])
#             #
#             # print('encpar_before', next(iter(graph_model.parameters())).grad[0,0])
#             # print('gpar_before', next(iter(graph_model.parameters()))[0,0])
#             graph_optimizer.step()
#         #     print('mu_aa',mu[0,0])
#         #     print('encpar_a', next(iter(graph_model.parameters())).grad[0,0])
#         #     print('gpar_a', next(iter(graph_model.parameters()))[0,0])
#         # # endGEC_optimizer.zero_grad()
#
#             # torch.cuda.empty_cache()
#
#             # if epoch % 10 == 0:
#             #     mu = update_mu(final_latent, device)
#
#             # train_losses.append(loss.item())
#             # print(loss.item(), np.mean(loss.item()))
#
#         # val_losses = []
#         # val_losses_enc = []
#         # model = model.eval()
#         # enc = enc.eval()
#         # with torch.no_grad():
#         #     for bat_num, batch_input in enumerate(val_dataset):
#         #         splitted_batch = np.split(batch_input, pmu_number, axis=-1)
#         #
#         #         for count, pmu_event_data in enumerate(splitted_batch):
#         #             pmu_event_data = pmu_event_data.to(device)
#         #             pred = model(pmu_event_data)
#         #             loss = criterion(pred.float(), pmu_event_data.float())
#         #             val_losses.append(loss.item())
#         #             # print(loss.item())
#         #
#         # train_loss = np.mean(train_losses)
#         # val_loss = np.mean(val_losses)
#         # history['train'].append(train_loss)
#         # history['val'].append(val_loss)
#         # if val_loss < best_loss:
#         #     best_loss = val_loss
#         #     best_model_wts = copy.deepcopy(model.state_dict())
#         #     # model_path = 'models/AED/806_824_836_846_graph_AED'
#         #     # torch.save(model, model_path)
#         #     # model_path = 'models/AED/806_824_836_846_graph_graph'
#         #     # torch.save(graph_model, model_path)
#         # # print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}  val loss enc {val_loss_enc}')
#         # print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
#     model.load_state_dict(best_model_wts)
#     return model.eval(), graph_model.eval()
#
#
# model, graph_model = train_model(
#     model, graph_model, train_dataloader, test_dataloader,
#     pmu_number=pmu_number, n_epochs=20
# )
#
# # #%%
# #
# # model_path = 'models/AED/806_824_836_846_graph_AED'
# # model = torch.load(model_path)
# # model.eval()
# #
# # model_path = 'models/AED/806_824_836_846_graph_graph'
# # graph_model = torch.load(model_path)
# # graph_model.eval()
# #
#%%
labels_cat = pd.DataFrame({'labels': labels})
labels_cat = labels_cat['labels'].astype('category').cat.codes.to_numpy()
lab_cat = np.unique(labels)
all_data = torch.from_numpy(per_unit).to(device)
# # get the latent variables
# selected_events = all_data[test_selector]
# selected_labels = labels_cat[test_selector]

selected_events = all_data[train_selector]
selected_labels = labels_cat[train_selector]


src = torch.from_numpy(np.array([0, 0, 0]))
dst = torch.from_numpy(np.array([1, 2, 3]))
enc = model_lat.encoder
model = model_lat.eval()
enc = enc.eval()
graph_model = graph_model.eval()

# selected_latent = model.encoder(all_data[train_selector[0:2000]]).cpu().detach().numpy()
# selected_labels = new_labels[train_selector[0:2000]]
all_latents = []
all_z = []
for idx, data in enumerate(selected_events):
    splitted_event_pmu = np.split(data, pmu_number, axis=-1)
    # all_pmu_latents = []
    for count, pmu_event_data in enumerate(splitted_event_pmu):

        pmu_event_data = pmu_event_data.to(device).reshape(1,125,9)
        pmu_batch_latent = enc(pmu_event_data)
        if count == 0:
            all_pmu_latents = pmu_batch_latent
        else:
            all_pmu_latents = torch.cat((all_pmu_latents, pmu_batch_latent))
    # print(all_pmu_latents.shape)
    all_z.append(all_pmu_latents.detach().cpu().numpy().ravel())
    all_pmu_latents = all_pmu_latents.clone().reshape(pmu_number, pmu_batch_latent.shape[0], pmu_batch_latent.shape[1])

    batched_graph = []
    # make graph data
    for idx in range(all_pmu_latents.shape[1]):  # iter on data in each batch
        node_features = all_pmu_latents[:, idx].clone()
        # Create a graph and add it to the list of graphs and labels.
        g = dgl.graph((src, dst), num_nodes=pmu_number).to(device)
        g.ndata['features'] = node_features
        g = dgl.add_self_loop(g)
        batched_graph.append(g)
    batched_graph = dgl.batch(batched_graph)

    # graph update
    final_latent = graph_model(batched_graph, batched_graph.ndata['features'].float())
    final_size = final_latent.shape
    final_latent = final_latent.reshape(final_size[0], final_size[-1]).clone()
    all_latents.append(final_latent.detach().cpu().numpy())

all_latents = np.array(all_latents)
all_z = np.array(all_z)

all_latents = all_latents.reshape(all_latents.shape[0],all_latents.shape[-1])
def all_clustering_models(latent, labels, cluster_num):
  from sklearn import metrics
  from sklearn.mixture import GaussianMixture
  from sklearn.cluster import AgglomerativeClustering

  #gmm
  pred_labels = GaussianMixture(n_components=cluster_num, random_state=0).fit_predict(latent)
  print('trian accuracy (ARS) for gmm', metrics.adjusted_rand_score(labels, pred_labels))

  #AgglomerativeClustering
  pred_labels = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(latent)
  print('trian accuracy (ARS) for AgglomerativeClustering', metrics.adjusted_rand_score(labels, pred_labels))

  from sklearn.cluster import DBSCAN
  pred_labels = DBSCAN().fit_predict(latent)
  print('trian accuracy (ARS) for DBSCAN', metrics.adjusted_rand_score(labels, pred_labels))

  from sklearn.cluster import KMeans
  pred_labels = KMeans(n_clusters=cluster_num, random_state=0).fit_predict(latent)
  print('trian accuracy (ARS) for KMeans', metrics.adjusted_rand_score(labels, pred_labels))

  # from sklearn.cluster import SpectralClustering
  # pred_labels = SpectralClustering(n_clusters=cluster_num, assign_labels="discretize", random_state=0).fit_predict(latent)
  # print('trian accuracy (ARS) for SpectralClustering', metrics.adjusted_rand_score(labels, pred_labels))

cluster_num = 9
all_clustering_models(all_latents, selected_labels, cluster_num)
all_clustering_models(all_z, selected_labels, cluster_num)

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(all_latents)

# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=selected_labels)
# plt.show()

colors = {'capbank840': 'darkgreen', 'capbank848':'lime', 'faultAB862':'hotpink', 'faultABC816': 'crimson',
       'faultC852':'gold', 'loada836':'cyan', 'motormed812':'dodgerblue', 'motorsmall828':'navy',
       'onephase858':'blueviolet'}
fig, ax = plt.subplots()
for ev in np.unique(selected_labels):
    # print(ev)
    ix = np.where(selected_labels == ev)

    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label = lab_cat[ev], s = 100)
ax.legend()
plt.show()


X_embedded = TSNE(n_components=2).fit_transform(all_z)

# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=selected_labels)
# plt.show()

colors = {'capbank840': 'darkgreen', 'capbank848':'lime', 'faultAB862':'hotpink', 'faultABC816': 'crimson',
       'faultC852':'gold', 'loada836':'cyan', 'motormed812':'dodgerblue', 'motorsmall828':'navy',
       'onephase858':'blueviolet'}
fig, ax = plt.subplots()
for ev in np.unique(selected_labels):
    # print(ev)
    ix = np.where(selected_labels == ev)

    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label = lab_cat[ev], s = 100)
ax.legend()
plt.show()
#%%


#%%
# model_path = 'models/AED/806_824_836_846_separate'
# torch.save(model, model_path)
#%%
def show_detail(data, pmu, type):

  fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, constrained_layout=True)
  for k in range(3):
    ax0.plot(data[5:,pmu * k])
    ax1.plot(data[5:,pmu * k + 3])
    ax2.plot(data[5:,pmu * k + 6])

  ax0.set_xlabel('time [s]')
  ax0.set_ylabel('voltage magnitude')
  ax0.legend(['v1', 'v2', 'v3'])

  ax1.set_xlabel('freq')
  ax1.set_ylabel('current magnitude')
  ax1.legend(['i1', 'i2', 'i3'])

  ax2.set_xlabel('time [s]')
  ax2.set_ylabel('angle diff')
  ax2.legend(['t1', 't2', 't3'])

  fig.title = 'real'
  if type == 'pred':
    fig.title = 'pred'

  return fig
#%%

pmu = 1
for ev in [102,483]:
# ev = 100
  data = new_per_unit[ev]
  data = torch.from_numpy(data).to(device).reshape(1, data.shape[0], data.shape[1])
  pred = model(data)
  def torch_to_numpy_cpu(data):
    return data.cpu()[0].detach().numpy()

  data = torch_to_numpy_cpu(data)
  pred = torch_to_numpy_cpu(pred)

  fig1 = show_detail(data, pmu, 'real')
  plt.show()
  fig2 = show_detail(pred, pmu, 'pred')
  plt.show()
#%%
def update_mu(final_latent, device):
    final_latent = final_latent.detach().cpu()
    kmeans = KMeans(9, n_init=20)
    kmeans.fit(final_latent)
    mu = torch.from_numpy(kmeans.cluster_centers_).to(device)
    mu = Variable(mu, requires_grad=True).to(device)
    return mu
batch_input = torch.ones((2,125,36), requires_grad = True)
splitted_batch = np.split(batch_input, pmu_number, axis=-1)
enc = model.encoder
dec = model.decoder
model = model.train()
enc = enc.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
enc_optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-5)
dec_optimizer = torch.optim.Adam(dec.parameters(), lr=1e-3, weight_decay=1e-5)
graph_optimizer = torch.optim.Adam(graph_model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='mean')
enc_criteria = LatentMu.apply
src = torch.from_numpy(np.array([0, 0, 0]))
dst = torch.from_numpy(np.array([1, 2, 3]))
# all_pmu_latents = []
initial =True
optimizer.zero_grad()
for count, pmu_event_data in enumerate(splitted_batch):
    pmu_event_data = pmu_event_data.to(device)
    # train AED with reconstruction loss
    predicted = model(pmu_event_data)
    loss = criterion(predicted.float(), pmu_event_data.float())
    # print('batch_grad', batch_input.grad[0,0][0])
    loss.backward()
    print('batch_grad', batch_input.grad[0, 0][0])
    # print('z grad', pmu_event_data.grad[0,0][0])
    pmu_batch_latent = enc(pmu_event_data)
    if count == 0:
        all_pmu_latents = pmu_batch_latent
    else:
        all_pmu_latents = torch.cat((all_pmu_latents, pmu_batch_latent))

optimizer.step()
for count, pmu_event_data in enumerate(splitted_batch):
    pmu_event_data = pmu_event_data.to(device)
    # train AED with reconstruction loss
    predicted = model(pmu_event_data)
    loss = criterion(predicted.float(), pmu_event_data.float())
    # print('batch_grad', batch_input.grad[0,0][0])
    loss.backward()
    print('batch_grad', batch_input.grad[0, 0][0])
    # print('z grad', pmu_event_data.grad[0,0][0])
    pmu_batch_latent = enc(pmu_event_data)
    if count == 0:
        new_all_pmu_latents = pmu_batch_latent
    else:
        new_all_pmu_latents = torch.cat((new_all_pmu_latents, pmu_batch_latent))
#%%
all_pmu_latents = all_pmu_latents.clone().reshape(pmu_number, pmu_batch_latent.shape[0],
                                                              pmu_batch_latent.shape[1])
batched_graph = []
# make graph data
for idx in range(all_pmu_latents.shape[1]):  # iter on data in each batch
    node_features = all_pmu_latents[:, idx].clone()
    # Create a graph and add it to the list of graphs and labels.
    g = dgl.graph((src, dst), num_nodes=pmu_number).to(device)
    g.ndata['features'] = node_features
    g = dgl.add_self_loop(g)
    batched_graph.append(g)
batched_graph = dgl.batch(batched_graph)
# print('before graph:', node_features)

# graph update
final_latent = graph_model(batched_graph, batched_graph.ndata['features'].float())
final_size = final_latent.shape
final_latent = final_latent.reshape(final_size[0], final_size[-1]).clone()
if initial:
    mu = update_mu(final_latent, device)
    initial = False
torch.autograd.set_detect_anomaly(True)
graph_loss = enc_criteria(final_latent, mu)
graph_optimizer.zero_grad()
graph_loss.backward()
print('batch_grad', batch_input.grad[0,0][0])
print('z grad', pmu_event_data.grad[0,0][0])
# print('z grad', pmu_event_data.grad[0,0][0])
# print('after graph:', final_latent)
# graph_optimizer.step()
# optimizer.step()
#%%
y = torch.sum(a**2)
print(a, y)
y.backward()
print(a.grad.data)
b = a.clone()
# b = b.reshape((3, 2))

g= torch.sum(b**3)
print(b,a,g)
g.backward()
print(a.grad.data)
#%%
d = next(iter(train_dataloader))
src = torch.from_numpy(np.array([0, 0, 0]))
dst = torch.from_numpy(np.array([1, 2, 3]))
enocdergraph = EncGraph(model.encoder, graph_model, pmu_number, src, dst)
#%%
encout = model.encoder(torch.ones((2, 125, 9)).to(device))
output = enocdergraph(d)
print(output[0,0], encout[0, 0])
optimizer = torch.optim.Adam(enocdergraph.parameters(), lr=1e-3, weight_decay=1e-5)
optimizer.zero_grad()
criterion = nn.MSELoss(reduction='mean')
l = criterion(output, torch.ones((100,8)).to(device))
l.backward()
optimizer.step()
#%%

ls = LatentMu.apply()
