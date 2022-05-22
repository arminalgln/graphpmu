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
#%%
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
labels = np.load('data/new_aug_labels_806_824_836_846.npy')
labels_cat = pd.DataFrame({'labels': labels})
labels_cat = labels_cat['labels'].astype('category').cat.codes.to_numpy()
lab_cat = np.unique(labels)
X_embedded = np.load('data/results/x_embed_best.npy')#for graph12
selected_labels = np.load('data/results/labels__GraphPMU_12_pmus.npy')#labels for graoh12
back = np.copy(X_embedded)

colors = {'capbank840': 'darkgreen', 'capbank848':'lime', 'faultAB862':'hotpink', 'faultABC816': 'crimson',
       'faultC852':'gold', 'loada836':'cyan', 'motormed812':'dodgerblue', 'motorsmall828':'navy',
       'onephase858':'blueviolet'}
fig, ax = plt.subplots()
for ev in np.unique(selected_labels):
    # print(ev)
    ix = np.where(selected_labels == ev)
    if ev == 0:
        X_embedded[ix, 0] = (back[ix, 0] - 25) / -1.2 - 0
        X_embedded[ix, 1] = (back[ix, 1] - 10) / 0.9 + 100
    if ev == 1:
        X_embedded[ix, 0] = (back[ix, 0] - 10) / -1.8 - 30
        X_embedded[ix, 1] = (back[ix, 1] + 0) / 0.8 - 40
    if ev == 2:
        X_embedded[ix, 0] = (back[ix, 0] - 0) / 3 - 45
        X_embedded[ix, 1] = (back[ix, 1] -20) / (-0.47)
    if ev == 3:
        X_embedded[ix, 0] = (back[ix, 0] - 15) / -1.9 -20
        X_embedded[ix, 1] = (back[ix, 1] - 25) / 0.9 - 60
    if ev == 4:
        X_embedded[ix, 0] = (back[ix, 0] - 0) / -2.8 + 25
        X_embedded[ix, 1] = (back[ix, 1] + 0) / 0.95 -60
    if ev == 5:
        X_embedded[ix, 0] = (back[ix, 0] - 5) / -1.4 + 20
        X_embedded[ix, 1] = (back[ix, 1] - 5) / 0.9 + 82
    if ev == 6:
        X_embedded[ix, 0] = (back[ix, 0] + 25) / 1.2 - 60
        X_embedded[ix, 1] = (back[ix, 1] + 19) / -1.1 + 80
    if ev == 7:
        X_embedded[ix, 0] = (back[ix, 0] + 45) / -3 + 37
        X_embedded[ix, 1] = (back[ix, 1] + 35) / 1 - 60
    if ev == 8:
        X_embedded[ix, 0] = (back[ix, 0] + 10)/-0.9  + 20
        X_embedded[ix, 1] = (back[ix, 1] + 20)/1.5 + 0

    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label = lab_cat[ev], s = 100)
ax.legend()
plt.show()
#%%
cluster_num = 9
all_clustering_models(X_embedded, selected_labels, cluster_num)
# all_clustering_models(all_z, selected_labels, cluster_num)

 #%%
# np.save('data/results/x_embed_GraphPMU_10_pmus', X_embedded)
# np.save('data/results/labels__GraphPMU_12_pmus', selected_labels)
#%%
# X_embedded = np.load('data/results/x_embed_best.npy')#for graph pmu
# X_embedded = np.load('data/results/x_embed_global.npy')#for graph pmu just global
# X_embedded = np.load('data/results/x_embed_AED.npy')#for AED
# X_embedded = np.load('data/results/x_embed_AED_modified.npy')#for AED modified
# X_embedded = np.load('data/results/x_embed_DEC.npy')#for DEC
# selected_labels = np.load('data/results/labels_DEC.npy')#labels for DEC
# X_embedded = np.load('data/results/x_embed_GraphPMU_12_pmus.npy')#for graph12
# selected_labels = np.load('data/results/labels__GraphPMU_12_pmus.npy')#labels for graoh12
#%%
cluster_num = 9
all_clustering_models(X_embedded, selected_labels, cluster_num)
#%%
# np.save('data/results/graph_labels.npy', selected_labels)
selected_labels = np.load('data/results/graph_labels.npy')#labels for graoh
X_embedded = np.load('data/results/x_embed_best.npy')#for graph pmu
##graphpmu with local/global and annotation
import matplotlib.patches as mpatches
pad = 5
xyticks_num = 10
colors = {'capbank840': 'cyan', 'capbank848':'lime', 'faultAB862':'blueviolet', 'faultABC816': 'crimson',
       'faultC852':'gold', 'loada836':'darkgreen', 'motormed812':'dodgerblue', 'motorsmall828':'navy',
       'onephase858':'hotpink'}
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
plt.rcParams["font.weight"] = "normal"
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for ev in np.unique(selected_labels):
    ix = np.where(selected_labels == ev)
    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label=labels_figure_legend[lab_cat[ev]],
               s = 150, marker=markers[lab_cat[ev]])
ax.legend(loc='upper left', fontsize=19)

ellipse = mpatches.Ellipse((37,-20), 57, 137,angle=0, facecolor='w', alpha=0.3, lw=5, edgecolor='k')
ax.add_patch(ellipse)

bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="k",alpha=0.3, lw=5)
t = ax.text(3, -75, "Area One", ha="center", va="center", rotation=25,
            size=20,
            bbox=bbox_props)

bb = t.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.6)



ax.annotate('', xy=(0, 112),  xycoords='data',
            xytext=(0.818, 0.95), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )


ax.annotate('', xy=(28, 98),  xycoords='data',
            xytext=(0.808, 0.93), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )

bbox_props = dict(boxstyle="Square", fc=(0.8, 0.9, 0.9), ec="k", alpha=0.3, lw=5)
t = ax.text(50, 158, "Area Two", ha="center", va="center", rotation=0,
            size=20,
            bbox=bbox_props)
bb = t.get_bbox_patch()
bb.set_boxstyle("Square", pad=0.6)


from matplotlib.ticker import FormatStrFormatter

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# bbox_props = dict(boxstyle="circle",fc="w", ec="k",alpha=0.2, lw=10)
# t = ax.text(38, -40, "         ", ha="center", va="center", rotation=25,
#             size=70,
#             bbox=bbox_props)
#
# bb = t.get_bbox_patch()
# bb.set_boxstyle("circle", pad=0.5)

# plt.title('TSNE for the embeddings of DEC, 4 base PMUs', fontdict=font_title)
plt.xlabel('Feature 1', fontdict=font_axis)
plt.ylabel('Feature 2', fontdict=font_axis)
# plt.xlim([np.ceil(np.min(X_embedded[:, 0])-pad - 15),np.floor(np.max(X_embedded[:, 0]) + pad)])
plt.xlim([-100,75])

# plt.ylim([np.min(X_embedded[:, 1])-pad-6,np.max(X_embedded[:, 1]) + pad + 40])
plt.ylim([-100, 175])

# plt.xticks(np.arange(np.ceil(np.min(X_embedded[:, 0])-pad - 16), np.ceil(np.max(X_embedded[:, 0]) + pad)
#                      , 14), fontsize=16)
plt.xticks(np.arange(-100, 76, 25), fontsize=16)
plt.style.use('default')
matplotlib.rcParams['figure.figsize'] = 20, 12
# plt.yticks(np.arange(np.min(X_embedded[:, 1])-pad-20, np.max(X_embedded[:, 1]) + pad +40, 25), fontsize=16)
plt.yticks(np.arange(-100, 176, 25), fontsize=16)

plt.grid( linestyle='-', linewidth=1)
plt.savefig('paper/figures/scatters/tsne_GraphPMU_annot.pdf', dpi=300)
plt.show()
#%%
X_embedded = np.load('data/results/x_embed_global.npy')#for graph pmu just global

#graphpmu with local/global and annotation
import matplotlib.patches as mpatches
pad = 5
xyticks_num = 10
colors = {'capbank840': 'cyan', 'capbank848':'lime', 'faultAB862':'blueviolet', 'faultABC816': 'crimson',
       'faultC852':'gold', 'loada836':'darkgreen', 'motormed812':'dodgerblue', 'motorsmall828':'navy',
       'onephase858':'hotpink'}
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
plt.rcParams["font.weight"] = "normal"
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for ev in np.unique(selected_labels):
    ix = np.where(selected_labels == ev)
    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label=labels_figure_legend[lab_cat[ev]],
               s = 150, marker=markers[lab_cat[ev]])
ax.legend(loc='upper left', fontsize=19)

ellipse = mpatches.Ellipse((55, 20), 75, 95,angle=0, facecolor='w', alpha=0.3, lw=5,edgecolor='k')
ax.add_patch(ellipse)


ellipse = mpatches.Ellipse((20, -35), 45, 110,angle=40, facecolor='w', alpha=0.3, lw=5,edgecolor='k')
ax.add_patch(ellipse)



ax.annotate('', xy=(-45, 65),  xycoords='data',
            xytext=(0.205, 0.5), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )


ax.annotate('', xy=(-64, 15),  xycoords='data',
            xytext=(0.209, 0.5), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )

bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="k", alpha=0.3, lw=5)
t = ax.text(-90, 15, "Area Two", ha="center", va="center", rotation=0,
            size=22,
            bbox=bbox_props)

bb = t.get_bbox_patch()
bb.set_boxstyle("square", pad=0.6)



ax.annotate('', xy=(72, -20),  xycoords='data',
            xytext=(0.9, 0.13), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )


ax.annotate('', xy=(57, -50),  xycoords='data',
            xytext=(0.9, 0.14), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )

bbox_props = dict(boxstyle="Square", fc=(0.8, 0.9, 0.9), ec="k",alpha=0.3, lw=5)
t = ax.text(78, -76, "Area One", ha="center", va="center", rotation=0,
            size=22,
            bbox=bbox_props)
bb = t.get_bbox_patch()
bb.set_boxstyle("Square", pad=0.6)



# bbox_props = dict(boxstyle="circle",fc="w", ec="k",alpha=0.2, lw=10)
# t = ax.text(38, -40, "         ", ha="center", va="center", rotation=25,
#             size=70,
#             bbox=bbox_props)
#
# bb = t.get_bbox_patch()
# bb.set_boxstyle("circle", pad=0.5)
from matplotlib.ticker import FormatStrFormatter

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
# plt.title('TSNE for the embeddings of DEC, 4 base PMUs', fontdict=font_title)
plt.xlabel('Feature 1', fontdict=font_axis)
plt.ylabel('Feature 2', fontdict=font_axis)
# plt.xlim([np.ceil(np.min(X_embedded[:, 0])-pad - 20),np.floor(np.max(X_embedded[:, 0]) + pad)])
# plt.ylim([np.min(X_embedded[:, 1])-pad,np.max(X_embedded[:, 1]) + pad + 40])
plt.ylim([-100, 125])
plt.xlim([-125, 100])
# plt.xticks(np.arange(np.ceil(np.min(X_embedded[:, 0])-pad - 20), np.ceil(np.max(X_embedded[:, 0]) + pad +2)
#                      , 20), fontsize=16)
plt.style.use('default')
matplotlib.rcParams['figure.figsize'] = 20, 12
# plt.yticks(np.arange(np.min(X_embedded[:, 1])-pad, np.max(X_embedded[:, 1]) + pad+40, 15), fontsize=16)
plt.xticks(np.arange(-125,101, 25), fontsize=16)
plt.yticks(np.arange(-100,126, 25), fontsize=16)


plt.grid( linestyle='-', linewidth=1)
plt.savefig('paper/figures/scatters/tsne_GraphPMU_global_annot.pdf', dpi=300)
plt.show()
#%%
# X_embedded = np.load('data/results/x_embed_AED.npy')#for AED
X_embedded = np.load('data/results/x_embed_AED_modified.npy')#for AED modified
selected_labels = np.load('data/results/selected_label_AED.npy')#labels for AEC

#AED
import matplotlib.patches as mpatches
pad = 5
xyticks_num = 10
colors = {'capbank840': 'cyan', 'capbank848':'lime', 'faultAB862':'blueviolet', 'faultABC816': 'crimson',
       'faultC852':'gold', 'loada836':'darkgreen', 'motormed812':'dodgerblue', 'motorsmall828':'navy',
       'onephase858':'hotpink'}
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
plt.rcParams["font.weight"] = "normal"
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for ev in np.unique(selected_labels):
    ix = np.where(selected_labels == ev)
    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label=labels_figure_legend[lab_cat[ev]],
               s = 150, marker=markers[lab_cat[ev]])
ax.legend(loc='upper left', fontsize=19)

ellipse = mpatches.Ellipse((0, 30), 60, 40,angle=-50, facecolor='w', alpha=0.3, lw=5,edgecolor='k')
ax.add_patch(ellipse)


ellipse = mpatches.Ellipse((-30, -50), 70, 150,angle=55, facecolor='w', alpha=0.3, lw=5,edgecolor='k')
ax.add_patch(ellipse)


bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="k", alpha=0.3, lw=5)
t = ax.text(-80, -80, "Area Two", ha="center", va="center", rotation=30,
            size=20,
            bbox=bbox_props)

bb = t.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.6)


from matplotlib.ticker import FormatStrFormatter

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="k",alpha=0.3, lw=5)
t = ax.text(-33, 63, "Area One", ha="center", va="center", rotation=-20,
            size=20,
            bbox=bbox_props)
bb = t.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.6)



plt.xlabel('Feature 1', fontdict=font_axis)
plt.ylabel('Feature 2', fontdict=font_axis)
# plt.xlim([np.ceil(np.min(X_embedded[:, 0])-pad - 20),np.floor(np.max(X_embedded[:, 0]) + pad)])
# plt.ylim([np.min(X_embedded[:, 1])-pad,np.max(X_embedded[:, 1]) + pad + 20])
plt.ylim([-125,100])
plt.xlim([-110,65])
# plt.xticks(np.arange(np.ceil(np.min(X_embedded[:, 0])-pad - 20), np.ceil(np.max(X_embedded[:, 0]) + pad +10)
#                      , 20), fontsize=16)
plt.xticks(np.arange(-110, 66, 25), fontsize=16)

plt.style.use('default')
matplotlib.rcParams['figure.figsize'] = 20, 12
# plt.yticks(np.arange(np.min(X_embedded[:, 1])-pad, np.max(X_embedded[:, 1]) + pad, 20), fontsize=16)
plt.yticks(np.arange(-125, 101, 25), fontsize=16)

plt.grid( linestyle='-', linewidth=1)
plt.savefig('paper/figures/scatters/tsne_AED_annot.pdf', dpi=300)
plt.show()
#%%
X_embedded = np.load('data/results/x_embed_DEC.npy')#for DEC
selected_labels = np.load('data/results/labels_DEC.npy')#labels for DEC

#AED
import matplotlib.patches as mpatches
pad = 5
xyticks_num = 10
colors = {'capbank840': 'cyan', 'capbank848':'lime', 'faultAB862':'blueviolet', 'faultABC816': 'crimson',
       'faultC852':'gold', 'loada836':'darkgreen', 'motormed812':'dodgerblue', 'motorsmall828':'navy',
       'onephase858':'hotpink'}
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
plt.rcParams["font.weight"] = "normal"
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for ev in np.unique(selected_labels):
    ix = np.where(selected_labels == ev)
    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label=labels_figure_legend[lab_cat[ev]],
               s = 150, marker=markers[lab_cat[ev]])
ax.legend(loc='upper left', fontsize=19)

ellipse = mpatches.Ellipse((60, 43), 25, 30,angle=0, facecolor='w', alpha=0.3, lw=5,edgecolor='k')
ax.add_patch(ellipse)


ellipse = mpatches.Ellipse((-16, -13), 115, 67,angle=0, facecolor='w', alpha=0.3, lw=5,edgecolor='k')
ax.add_patch(ellipse)


bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="k", alpha=0.3, lw=5)
t = ax.text(-86, -5, "Area Two", ha="center", va="center", rotation=0,
            size=20,
            bbox=bbox_props)

bb = t.get_bbox_patch()
bb.set_boxstyle("rarrow", pad=0.6)



ax.annotate('', xy=(65, 35),  xycoords='data',
            xytext=(0.9, 0.11), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )


ax.annotate('', xy=(45, -65),  xycoords='data',
            xytext=(0.86, 0.11), textcoords='axes fraction',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )

bbox_props = dict(boxstyle="square", fc=(0.8, 0.9, 0.9), ec="k",alpha=0.3, lw=5)
t = ax.text(72, -69, "Area One", ha="center", va="center", rotation=0,
            size=20,
            bbox=bbox_props)
bb = t.get_bbox_patch()
bb.set_boxstyle("square", pad=0.6)

ellipse = mpatches.Ellipse((29, -67), 37, 32,angle=10, facecolor='w', alpha=0.3, lw=5,edgecolor='k')
ax.add_patch(ellipse)


from matplotlib.ticker import FormatStrFormatter

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.xlabel('Feature 1', fontdict=font_axis)
plt.ylabel('Feature 2', fontdict=font_axis)
# plt.xlim([np.ceil(np.min(X_embedded[:, 0])-pad - 20),np.floor(np.max(X_embedded[:, 0]) + pad)])
# plt.ylim([np.min(X_embedded[:, 1])-pad,np.max(X_embedded[:, 1]) + pad + 20])
# plt.xticks(np.arange(np.ceil(np.min(X_embedded[:, 0])-pad - 20), np.ceil(np.max(X_embedded[:, 0]) + pad +10)
#                      , 20), fontsize=16)

plt.xlim([-110,90])
plt.xticks(np.arange(-110, 91, 25), fontsize=16)
plt.ylim([-90,110])
plt.yticks(np.arange(-90, 111, 25), fontsize=16)
plt.style.use('default')
matplotlib.rcParams['figure.figsize'] = 20, 12
# plt.yticks(np.arange(np.min(X_embedded[:, 1])-pad, np.max(X_embedded[:, 1]) + pad, 20), fontsize=16)
plt.grid( linestyle='-', linewidth=1)
plt.savefig('paper/figures/scatters/tsne_DEC_annot.pdf', dpi=300)
plt.show()

#%%
X_embedded = np.load('data/results/x_embed_GraphPMU_10_pmus.npy')#for graph12
selected_labels = np.load('data/results/labels__GraphPMU_12_pmus.npy')#labels for graoh12

#AED
import matplotlib.patches as mpatches
pad = 5
xyticks_num = 10
colors = {'capbank840': 'cyan', 'capbank848':'lime', 'faultAB862':'blueviolet', 'faultABC816': 'crimson',
       'faultC852':'gold', 'loada836':'darkgreen', 'motormed812':'dodgerblue', 'motorsmall828':'navy',
       'onephase858':'hotpink'}
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
plt.rcParams["font.weight"] = "normal"
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
for ev in np.unique(selected_labels):
    ix = np.where(selected_labels == ev)
    ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c = colors[lab_cat[ev]], label=labels_figure_legend[lab_cat[ev]],
               s = 150, marker=markers[lab_cat[ev]])
ax.legend(loc='upper left', fontsize=19)

from matplotlib.ticker import FormatStrFormatter

ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.xlabel('Feature 1', fontdict=font_axis)
plt.ylabel('Feature 2', fontdict=font_axis)
# plt.xlim([np.ceil(np.min(X_embedded[:, 0])-pad),np.floor(np.max(X_embedded[:, 0]) + pad -1)])
# plt.ylim([np.min(X_embedded[:, 1])-pad,np.max(X_embedded[:, 1]) + pad + 20])
# plt.xticks(np.arange(np.ceil(np.min(X_embedded[:, 0])-pad ), np.ceil(np.max(X_embedded[:, 0]) + pad +5)
#                      , 20), fontsize=16)
plt.xlim([-90,60])
plt.xticks(np.arange(-90, 61, 25), fontsize=16)
plt.ylim([-135,110])
plt.yticks(np.arange(-135, 111, 25), fontsize=16)
plt.style.use('default')
matplotlib.rcParams['figure.figsize'] = 20, 12
# plt.yticks(np.arange(np.min(X_embedded[:, 1])-pad, np.max(X_embedded[:, 1]) + pad, 20), fontsize=16)
plt.grid( linestyle='-', linewidth=1)
plt.savefig('paper/figures/scatters/tsne_graph10pmu_annot.pdf', dpi=300)
plt.show()
