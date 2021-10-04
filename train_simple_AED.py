# from sequitur.models import LSTM_AE
import torch
from torch import nn
import numpy as np
# from sequitur import quick_train
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from models.AED.simpleAED import Encoder, Decoder, RecurrentAutoencoder
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn, optim
import pandas as pd

import torch.nn.functional as F
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#load the data
per_unit = np.load('data/aug_all_per_unit_806_824_836_846.npy')
labels = np.load('data/aug_labels_806_824_836_846.npy')


# normalize data
new_data = []

for i in range(per_unit.shape[-1]):
  mx = np.max(per_unit[:, :, i], axis=1)
  # mn = np.min(per_unit[:, :, i], axis=1)

  new_data.append((per_unit[:, :, i])/(mx[:, None]))

new_data = np.array(new_data)
new_data = np.swapaxes(new_data, 0, 1)
per_unit = np.swapaxes(new_data, 2, 1)


n_seq, seq_len, n_features = per_unit.shape

num_examples = per_unit.shape[0]
num_train = int(num_examples * 0.9)
train_selector = np.random.choice(num_examples, num_train, replace=False)
test_selector = np.setdiff1d(np.arange(num_examples), train_selector)


train_sampler = SubsetRandomSampler(torch.from_numpy(train_selector))
test_sampler = SubsetRandomSampler(torch.from_numpy(test_selector))

b_size = 100

train_dataloader = DataLoader(
    per_unit, sampler=train_sampler, batch_size=b_size, drop_last=False)
test_dataloader = DataLoader(
    per_unit, sampler=test_sampler, batch_size=b_size, drop_last=False)
#%%
#define and train a model and save
model = RecurrentAutoencoder(125, 36, 32)
model.to(device)
model.float()

def train_model(model, train_dataset, val_dataset, n_epochs):

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = criterion = nn.MSELoss(reduction='mean')
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  for epoch in range(1, n_epochs + 1):
    model = model.train()
    train_losses = []
    for seq_true in train_dataset:
      seq_true = seq_true.to(device)
      optimizer.zero_grad()
      seq_pred = model(seq_true)
      loss = criterion(seq_pred.float(), seq_true.float())
      # print(epoch, loss)
      loss.backward()
      optimizer.step()
      train_losses.append(loss.item())
    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:
        seq_true = seq_true.to(device)
        seq_pred = model(seq_true)
        loss = criterion(seq_pred.float(), seq_true.float())
        val_losses.append(loss.item())
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    if val_loss < best_loss:
      best_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)
  return model.eval(), history
#%%
model, history = train_model(
  model,
  train_dataloader,
  test_dataloader,
  n_epochs=100
)
#%%
model_path = 'models/AED/806_824_836_846_stacked'
torch.save(model, model_path)
#%%
def show_detail(data, pmu, type):
  fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, constrained_layout=True)
  for k in range(3):
    ax0.plot(data[5:,pmu * k])
    ax1.plot(data[5:,pmu * k + 3])
    ax2.plot(data[5:,pmu * k + 6])

  ax0.set_xlabel('timesteps')
  ax0.set_ylabel('voltage magnitude')
  ax0.legend(['v1', 'v2', 'v3'])

  ax1.set_xlabel('timesteps')
  ax1.set_ylabel('current magnitude')
  ax1.legend(['i1', 'i2', 'i3'])

  ax2.set_xlabel('timesteps')
  ax2.set_ylabel('angle diff')
  ax2.legend(['t1', 't2', 't3'])

  fig.title = 'real'
  if type == 'pred':
    fig.title = 'pred'

  return fig
#%%

pmu = 1
for ev in range(5):
# ev = 100
  data = per_unit[ev]
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
#load the trained model
model_path = 'models/AED/806_824_836_846_stacked'
model = torch.load(model_path)
model.eval()
#%%
labels = pd.DataFrame({'labels': labels})
labels = labels['labels'].astype('category').cat.codes.to_numpy()
#%%
all_data = torch.from_numpy(per_unit).to(device)
# get the latent variables
#%%
selected_latent = model.encoder(all_data[train_selector[2387:3580]]).cpu().detach().numpy()
selected_labels = labels[train_selector[2387:3580]]
#%%

#clustering results based on different clustering models
#but the representation learning of the latent space is the important part not the clustering model
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

  from sklearn.cluster import SpectralClustering
  pred_labels = SpectralClustering(n_clusters=cluster_num, assign_labels="discretize", random_state=0).fit_predict(latent)
  print('trian accuracy (ARS) for SpectralClustering', metrics.adjusted_rand_score(labels, pred_labels))



cluster_num = 9
all_clustering_models(selected_latent, selected_labels, cluster_num)
#%%
#show TSNE of the clusters based on the selected latent
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(selected_latent)
from matplotlib.colors import ListedColormap
#%%
pad = 5
xyticks_num = 10
unique_labels = np.unique(selected_labels)
clrs = ['r','g','b','c','m','y','k','orange','lime']
values = [unique_labels.tolist().index(i) for i in selected_labels]
plt.style.use('default')
matplotlib.rcParams['figure.figsize'] = 20, 12
# colors = ListedColormap(['r','b','g'])
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=values, s=100, cmap='tab10')
plt.title('TSNE for the embeddings of stacked AED with DEC')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.xlim([np.min(X_embedded[:, 0])-pad,np.max(X_embedded[:, 0]) + pad])
plt.ylim([np.min(X_embedded[:, 1])-pad,np.max(X_embedded[:, 1]) + pad])
plt.xticks(np.arange(np.min(X_embedded[:, 0])-pad, np.max(X_embedded[:, 0]) + pad, 5))

plt.yticks(np.arange(np.min(X_embedded[:, 1])-pad, np.max(X_embedded[:, 1]) + pad, 5))
plt.grid()
plt.legend(handles=scatter.legend_elements()[0], labels=unique_labels.tolist(),scatterpoints=10, fontsize=20)
plt.tight_layout()
plt.savefig('figures/tsne_stacked_AED_DEC.png', dpi=300)
plt.show()
