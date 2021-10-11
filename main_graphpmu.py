import torch
import pickle
import numpy as np
from models.graph_model import GraphPMU, GraphEncoder
from models.discriminator import Discriminator
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import dgl
from dgl.dataloading import GraphDataLoader
import copy
from models.losses import global_loss_, get_positive_expectation, get_negative_expectation
import torch.nn as nn
import pandas as pd
#%%

with open('data/positive_graphs.pkl', 'rb') as handle:
  pos_graphs = pickle.load(handle)

with open('data/negative_graphs.pkl', 'rb') as handle:
  neg_graphs = pickle.load(handle)
#%%
#initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_encoder = GraphEncoder
disc = Discriminator
node_nums = pos_graphs[0].num_nodes()
[in_feats, h1_feats, last_space_feature] = [pos_graphs[0].ndata['features'].shape[-1], 64, 8]
[D_h1, D_h2] = [16, 32]
measure = 'JSD'#['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
graphpmu = GraphPMU(g_encoder, disc, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2, measure, device)

#make positive and negative batches
num_samples = len(pos_graphs)

num_train = int(num_samples * 0.9)
train_selector = np.random.choice(num_samples, num_train, replace=False)
test_selector = np.setdiff1d(np.arange(num_samples), train_selector)

train_index = np.random.choice(train_selector.shape[0], 10000, replace=False)
test_index = np.random.choice(test_selector.shape[0], 100, replace=False)

train_selector = train_selector[train_index]
test_selector = test_selector[test_index]


train_sampler = SubsetRandomSampler(torch.from_numpy(train_selector))
test_sampler = SubsetRandomSampler(torch.from_numpy(test_selector))

b_size = 100

pos_train_dataloader = GraphDataLoader(
    pos_graphs, sampler=train_sampler, batch_size=b_size, drop_last=False)
pos_test_dataloader = GraphDataLoader(
    pos_graphs, sampler=test_sampler, batch_size=b_size, drop_last=False)
neg_train_dataloader = GraphDataLoader(
    neg_graphs, sampler=train_sampler, batch_size=b_size, drop_last=False)
neg_test_dataloader = GraphDataLoader(
    neg_graphs, sampler=test_sampler, batch_size=b_size, drop_last=False)

#%%
def train_graphpmu(graphpmu, pos_train_dataloader, neg_train_dataloader,
                   pos_test_dataloader, neg_test_dataloader, epochs_num):
    #initialization for training
    graphpmu_optimizer = torch.optim.Adam(graphpmu.parameters(), lr=1e-3)#, weight_decay=1e-4
    criteria = global_loss_
    BCE = nn.BCELoss()
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(graphpmu.state_dict())
    best_loss = 10000.0


    for epoch in range(epochs_num):
        #training mode
        graphpmu = graphpmu.train()
        graphpmu_optimizer.zero_grad()
        train_losses = []

        pos_iter = iter(pos_train_dataloader)
        neg_iter = iter(neg_train_dataloader)
        for pos_batch in pos_iter:#iter on train batches
            pos_neg_graphs = dgl.batch([pos_batch, next(neg_iter)]) #concat pos and neg graphs
            pred = graphpmu(pos_neg_graphs)
            target = torch.cat((torch.ones(pos_batch.batch_size, device=device), torch.zeros(pos_batch.batch_size, device=device)))
            loss = BCE(pred.ravel(), target)
            # loss = criteria(pred, measure)
            # print(loss)
            graphpmu_optimizer.zero_grad()
            loss.backward()
            train_losses.append(loss.item())
            graphpmu_optimizer.step()

        validation_losses = []
        graphpmu = graphpmu.eval()
        with torch.no_grad():
            pos_iter = iter(pos_test_dataloader)
            neg_iter = iter(neg_test_dataloader)
            for pos_batch in pos_iter:  # iter on train batches
                pos_neg_graphs = dgl.batch([pos_batch, next(neg_iter)])  # concat pos and neg graphs
                pred = graphpmu(pos_neg_graphs)
                target = torch.cat(
                    (torch.ones(pos_batch.batch_size, device=device), torch.zeros(pos_batch.batch_size, device=device)))
                loss = BCE(pred.ravel(), target)
                # loss = criteria(pred, measure)
                validation_losses.append(loss.item())
                    # print(loss.item())

        train_loss = np.mean(train_losses)
        validation_loss = np.mean(validation_losses)
        history['train'].append(train_loss)
        history['val'].append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_model_wts = copy.deepcopy(graphpmu.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {validation_loss}')
    graphpmu.load_state_dict(best_model_wts)
    return graphpmu.eval()


gpmodel = train_graphpmu(graphpmu, pos_train_dataloader, neg_train_dataloader,
                         pos_test_dataloader, neg_test_dataloader, epochs_num=100)
#%%
from sklearn.metrics import accuracy_score
#discriminator evaluation
def get_accuracy(y_true, y_prob):
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)
with torch.no_grad():
    pos_iter = iter(pos_train_dataloader)
    neg_iter = iter(neg_train_dataloader)
    count = 0
    for pos_batch in pos_iter:  # iter on train batches
        print(count)
        count += 1
        pos_neg_graphs = dgl.batch([pos_batch, next(neg_iter)])  # concat pos and neg graphs
        g_enc = gpmodel.encoder(pos_neg_graphs)
        # pred = gpmodel.discriminator(g_enc)


        pred = gpmodel(pos_neg_graphs)
        y = torch.cat((torch.ones(pos_batch.batch_size), torch.zeros(pos_batch.batch_size)))
        target = torch.cat(
            (torch.ones(pos_batch.batch_size, device=device), torch.zeros(pos_batch.batch_size, device=device)))
        # loss = BCE(pred, target)
        print(get_accuracy(target, pred))

#%%

# clustering evaluation
labels = np.load('data/new_aug_labels_806_824_836_846.npy')
labels_cat = pd.DataFrame({'labels': labels})
labels_cat = labels_cat['labels'].astype('category').cat.codes.to_numpy()
lab_cat = np.unique(labels)

train_selector_partial = train_selector[0:1000]

selected_events = [pos_graphs[i] for i in train_selector_partial]
selected_labels = labels_cat[train_selector_partial]

gpmodel = gpmodel.eval()
all_latents = []
with torch.no_grad():
    for event_graph in selected_events:
        g_enc = gpmodel.encoder(event_graph)
        all_latents.append(g_enc.detach().cpu().numpy())
print(len(all_latents))
print(all_latents[0].shape)
all_latents = np.array(all_latents)
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
# all_clustering_models(all_z, selected_labels, cluster_num)

#%%
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(all_latents)
import matplotlib.pyplot as plt
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



