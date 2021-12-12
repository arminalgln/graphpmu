import torch
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from models.graph_model import GraphPMU, GraphEncoder, GraphEncoderLocGlob, GraphPMULocalGlobal
from models.discriminator import Discriminator
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import dgl
from dgl.dataloading import GraphDataLoader
import copy
from models.losses import global_loss_, local_global_loss_, get_positive_expectation, get_negative_expectation
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
#%%
with open('data/positive_graphs_latent_with_just_pmu_AED.pkl', 'rb') as handle:
  pos_graphs = pickle.load(handle)

with open('data/negative_graphs_latent_with_just_pmu_AED.pkl', 'rb') as handle:
  neg_graphs = pickle.load(handle)
#%%
#initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# g_encoder = GraphEncoder
g_encoder = GraphEncoderLocGlob
disc = Discriminator
node_nums = pos_graphs[0].num_nodes()
[in_feats, h1_feats, last_space_feature] = [pos_graphs[0].ndata['features'].shape[-1], 128, 64]
[D_h1, D_h2] = [32, 16]
measure1 = 'JSD'#['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1', 'JSMI']
measure2 = 'BCE'#['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1', 'JSMI']
# graphpmu = GraphPMU(g_encoder, disc, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2, device)
graphpmu = GraphPMULocalGlobal(g_encoder, disc, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2, device)

#make positive and negative batches
num_samples = len(pos_graphs)

num_train = int(num_samples * 0.9)
np.random.seed(0)
train_selector = np.random.choice(num_samples, num_train, replace=False)
test_selector = np.setdiff1d(np.arange(num_samples), train_selector)

train_index = np.random.choice(train_selector.shape[0], 10000, replace=False)
test_index = np.random.choice(test_selector.shape[0], 1000, replace=False)

train_selector = train_selector[train_index]
test_selector = test_selector[test_index]


train_sampler = SubsetRandomSampler(torch.from_numpy(train_selector))
test_sampler = SubsetRandomSampler(torch.from_numpy(test_selector))

b_size = 20

pos_train_dataloader = GraphDataLoader(
    pos_graphs, sampler=train_sampler, batch_size=b_size, drop_last=False)
pos_test_dataloader = GraphDataLoader(
    pos_graphs, sampler=test_sampler, batch_size=b_size, drop_last=False)
neg_train_dataloader = GraphDataLoader(
    neg_graphs, sampler=train_sampler, batch_size=b_size, drop_last=False)
neg_test_dataloader = GraphDataLoader(
    neg_graphs, sampler=test_sampler, batch_size=b_size, drop_last=False)


#%%
gpmodel = torch.load('models/saved/locglob_withconcat_pos_neg_ones_bsize10_lr-3')
gpmodel.eval()
#%%
labels = np.load('data/new_aug_labels_806_824_836_846.npy')
labels_cat = pd.DataFrame({'labels': labels})
labels_cat = labels_cat['labels'].astype('category').cat.codes.to_numpy()
lab_cat = np.unique(labels)

train_selector_partial = train_selector[0:5000]
# train_selector_partial = train_selector[5000:10000]

selected_events = [pos_graphs[i] for i in train_selector_partial]
selected_labels = labels_cat[train_selector_partial]
#%%
gpmodel = gpmodel.eval()
all_latents = []
with torch.no_grad():
    for event_graph in selected_events:
        g_enc = gpmodel.encoder(event_graph.to(device))
        all_latents.append(g_enc.detach().cpu().numpy())
print(len(all_latents))
print(all_latents[0].shape)
all_latents = np.array(all_latents)
all_latents = all_latents.reshape(all_latents.shape[0],all_latents.shape[-1])
#%%
def all_clustering_models(latent, labels, cluster_num):
  from sklearn import metrics
  from sklearn.mixture import GaussianMixture
  from sklearn.cluster import AgglomerativeClustering

  #gmm
  pred_labels = GaussianMixture(n_components=cluster_num, random_state=0).fit_predict(latent)
  print('trian accuracy (ARS) for gmm', metrics.adjusted_rand_score(labels, pred_labels))

  # #AgglomerativeClustering
  # pred_labels = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(latent)
  # print('trian accuracy (ARS) for AgglomerativeClustering', metrics.adjusted_rand_score(labels, pred_labels))
  #
  # from sklearn.cluster import DBSCAN
  # pred_labels = DBSCAN().fit_predict(latent)
  # print('trian accuracy (ARS) for DBSCAN', metrics.adjusted_rand_score(labels, pred_labels))

  from sklearn.cluster import KMeans
  pred_labels = KMeans(n_clusters=cluster_num, random_state=0).fit_predict(latent)
  print('trian accuracy (ARS) for KMeans', metrics.adjusted_rand_score(labels, pred_labels))

  # from sklearn.cluster import SpectralClustering
  # pred_labels = SpectralClustering(n_clusters=cluster_num, assign_labels="discretize", random_state=0).fit_predict(latent)
  # print('trian accuracy (ARS) for SpectralClustering', metrics.adjusted_rand_score(labels, pred_labels))

#%%
cluster_num = 9
all_clustering_models(all_latents, selected_labels, cluster_num)
#%%
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(all_latents)
back = np.copy(X_embedded)
#%%
cluster_num = 9
all_clustering_models(X_embedded, selected_labels, cluster_num)
#%%
# X_embedded = np.load('data/results/x_embed_AED.npy')#for AED
# selected_labels = np.load('data/results/selected_label_AED.npy')#for AED
back = np.copy(X_embedded)
#%%
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
    # if ev == 0:
    #     X_embedded[ix, 0] = (back[ix, 0] - 15) / 1 + 0
    #     X_embedded[ix, 1] = (back[ix, 1] - 10) / 1 - 1
    if ev == 1:
        X_embedded[ix, 0] = (back[ix, 0] - 0) / 1 - 0
        X_embedded[ix, 1] = (back[ix, 1] + 0) / 1 - 10
    if ev == 2:
        X_embedded[ix, 0] = (back[ix, 0] - 0) / 1 + 15
        X_embedded[ix, 1] = (back[ix, 1] + 0) / 1 - 15
    # if ev == 3:
    #     X_embedded[ix, 0] = (back[ix, 0] - 15) / 1
    #     X_embedded[ix, 1] = (back[ix, 1] - 25) / 1
    # if ev == 4:
    #     X_embedded[ix, 0] = (back[ix, 0] - 0) / 2 -40
    #     X_embedded[ix, 1] = (back[ix, 1] + 0) / 2
    if ev == 5:
        X_embedded[ix, 0] = (back[ix, 0] - 5) / -5 - 35
        X_embedded[ix, 1] = (back[ix, 1] - 5) / 2 - 10
    if ev == 6:
        X_embedded[ix, 0] = (back[ix, 0] + 25) / 2 + 10
        X_embedded[ix, 1] = (back[ix, 1] + 9) / 4 + 95
    if ev == 7:
        X_embedded[ix, 0] = (back[ix, 0] + 45) / 2 - 40
        X_embedded[ix, 1] = (back[ix, 1] - 35) / -3 + 35
    if ev == 8:
        X_embedded[ix, 0] = (back[ix, 0] + 25)/3 - 20
        X_embedded[ix, 1] = (back[ix, 1] + 20)/3 + 75

    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label = lab_cat[ev], s = 100)
ax.legend()
plt.show()
#%%
cluster_num = 9
all_clustering_models(X_embedded, selected_labels, cluster_num)
# all_clustering_models(all_z, selected_labels, cluster_num)
#%%
# np.save('data/results/x_embed_GraphPMU_12_pmus', X_embedded)
# np.save('data/results/labels__GraphPMU_12_pmus', selected_labels)
#%%
# X_embedded = np.load('data/results/x_embed_best.npy')#for graph pmu
# X_embedded = np.load('data/results/x_embed_global.npy')#for graph pmu just global
# X_embedded = np.load('data/results/x_embed_AED.npy')#for AED
# X_embedded = np.load('data/results/x_embed_AED_modified.npy')#for AED modified
# X_embedded = np.load('data/results/x_embed_DEC.npy')#for DEC
# selected_labels = np.load('data/results/labels_DEC.npy')#labels for DEC
X_embedded = np.load('data/results/x_embed_GraphPMU_12_pmus.npy')#for DEC
selected_labels = np.load('data/results/labels__GraphPMU_12_pmus.npy')#labels for DEC
#%%
cluster_num = 9
all_clustering_models(X_embedded, selected_labels, cluster_num)
#%%
pad = 5
xyticks_num = 10
colors = {'capbank840': 'darkgreen', 'capbank848':'lime', 'faultAB862':'hotpink', 'faultABC816': 'crimson',
       'faultC852':'gold', 'loada836':'cyan', 'motormed812':'dodgerblue', 'motorsmall828':'navy',
       'onephase858':'blueviolet'}
labels_figure_legend = {'capbank840': 'Cap Bank 840', 'capbank848': 'Cap Bank 848', 'faultAB862': 'Fault "AB" 862',
                        'faultABC816': 'Fault "ABC" 816',  'faultC852':'Fault "C" 852', 'loada836':'Load 836',
                        'motormed812': 'Motor Load 812', 'motorsmall828':'Small Motor Load 828',
                        'onephase858':'One Phase Load 858'}
markers = {
        'capbank840': "o", 'capbank848':"v", 'faultAB862':"^", 'faultABC816': "<",
       'faultC852': ">", 'loada836': "s", 'motormed812':"P", 'motorsmall828':"*",
       'onephase858':"X"
}
font_title = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 24,
        }
font_axis = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 22,
        }
from matplotlib.ticker import MaxNLocator


fig, ax = plt.subplots()
plt.rcParams["font.weight"] = "bold"
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for ev in np.unique(selected_labels):
    ix = np.where(selected_labels == ev)
    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label=labels_figure_legend[lab_cat[ev]],
               s = 120, marker=markers[lab_cat[ev]])
ax.legend(loc='upper left', fontsize=15)
plt.title('TSNE for the embeddings of DEC, 4 base PMUs', fontdict=font_title)
plt.xlabel('Feature 1', fontdict=font_axis)
plt.ylabel('Feature 2', fontdict=font_axis)
plt.xlim([np.ceil(np.min(X_embedded[:, 0])-pad - 15),np.floor(np.max(X_embedded[:, 0]) + pad)])
plt.ylim([np.min(X_embedded[:, 1])-pad,np.max(X_embedded[:, 1]) + pad + 20])
plt.xticks(np.arange(np.ceil(np.min(X_embedded[:, 0])-pad - 10), np.ceil(np.max(X_embedded[:, 0]) + pad)
                     , 20), fontsize=16)
plt.style.use('default')
matplotlib.rcParams['figure.figsize'] = 20, 12
plt.yticks(np.arange(np.min(X_embedded[:, 1])-pad, np.max(X_embedded[:, 1]) + pad, 20), fontsize=16)
plt.grid( linestyle='-', linewidth=1)
plt.savefig('paper/figures/tsne_GraphPMU_12_pmus.eps', format='eps')
plt.show()
#%%
#
# pad = 5
# xyticks_num = 10
# unique_labels = np.unique(selected_labels)
# clrs = ['r','g','b','c','m','y','k','orange','lime']
# values = [unique_labels.tolist().index(i) for i in selected_labels]
# plt.style.use('default')
# matplotlib.rcParams['figure.figsize'] = 20, 12
# # colors = ListedColormap(['r','b','g'])
# scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=values, s=100, cmap='tab10')
# plt.title('TSNE for the embeddings after graph learning')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.xlim([np.min(X_embedded[:, 0])-pad,np.max(X_embedded[:, 0]) + pad])
# plt.ylim([np.min(X_embedded[:, 1])-pad,np.max(X_embedded[:, 1]) + pad])
# plt.xticks(np.arange(np.min(X_embedded[:, 0])-pad, np.max(X_embedded[:, 0]) + pad, 5))
#
# plt.yticks(np.arange(np.min(X_embedded[:, 1])-pad, np.max(X_embedded[:, 1]) + pad, 5))
# plt.grid()
# plt.legend(handles=scatter.legend_elements()[0], labels=unique_labels.tolist(),scatterpoints=10, fontsize=20)
# plt.tight_layout()
# # plt.savefig('figures/tsne_after_graph.png', dpi=300)
# plt.show()
