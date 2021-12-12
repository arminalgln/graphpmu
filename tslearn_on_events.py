import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tslearn
import os
from tslearn.utils import to_time_series_dataset

from tslearn.clustering import KShape, TimeSeriesKMeans, KernelKMeans, silhouette_score
from tslearn.neural_network import TimeSeriesMLPClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax
from sklearn import metrics

# from sequitur.models import LSTM_AE
import torch
from torch import nn
import numpy as np
# from sequitur import quick_train
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from models.AED.simpleAED import Encoder, Decoder,RecurrentAutoencoder
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn, optim
import pandas as pd

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load the data
per_unit = np.load('data/new_aug_all_per_unit_806_824_836_846.npy')
labels = np.load('data/new_aug_labels_806_824_836_846.npy')


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


# features = np.load('data/features.npy')
# labels = np.load('data/labels.npy')
# data = np.load('data/all_event_data.npy')
# per_unit = np.load('data/all_per_unit.npy')
#

X_train = to_time_series_dataset(per_unit[test_selector])
y_train = np.array([np.where(np.unique(labels[test_selector]) == i)[0][0] for i in labels[test_selector]])
cluster_number = np.unique(y_train).shape[0]
print(cluster_number)
seed = 0
np.random.seed(seed)
X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)
ks = KShape(n_clusters=cluster_number, n_init=1, verbose=True, random_state=seed)
y_pred_kshape = ks.fit_predict(X_train, y_train)
print('kshape finished')
#%%
model = TimeSeriesKMeans(n_clusters=cluster_number, metric="softdtw",
                         max_iter=2, random_state=seed)
y_pred_km = model.fit_predict(X_train)
print('km finished')

#%%
gak_km = KernelKMeans(n_clusters=cluster_number,
                      kernel="gak",
                      kernel_params={"sigma": "auto"},
                      n_init=20,
                      verbose=True,
                      random_state=seed)
y_pred_gak_km = gak_km.fit_predict(X_train)
#%%
accuracy = {'kshape_Tslearn': 0.2371, 'kmeans_Tslearn': 0.3435, 'kernel_Tslearn': 0.4181,
            'GMM_AED': 0.532283, 'GMM_DEC_AED': 0.55777}
print('rand score')
print(metrics.adjusted_rand_score(y_train, y_pred_km))
print(metrics.adjusted_rand_score(y_train, y_pred_gak_km))
print(metrics.adjusted_rand_score(y_train, y_pred_kshape))
#%%
def show_one_event(X_train, id, pmu):
    event = X_train[id]
    plt.figure()

    shift = pmu * 9
    subfig = 311
    for f in range(3):
        plt.subplot(subfig)
        for i in range(f*3, (f+1)*3):
            plt.plot(event[:, shift + i])
        subfig += 1
    plt.show()
#%%
print('rand score')
print(metrics.adjusted_rand_score(y_train, y_pred_km))
print(metrics.adjusted_rand_score(y_train, y_pred_gak_km))
print(metrics.adjusted_rand_score(y_train, y_pred_kshape))
# print(metrics.adjusted_rand_score(y_train, yy))
print('\n-----------mutual info score--------------\n')
print(metrics.adjusted_mutual_info_score(y_train, y_pred_km))
print(metrics.adjusted_mutual_info_score(y_train, y_pred_gak_km))
print(metrics.adjusted_mutual_info_score(y_train, y_pred_kshape))
# print(metrics.adjusted_mutual_info_score(y_train, yy))
print('\n-------------------v measure score------------------\n')

print(metrics.v_measure_score(y_train, y_pred_km))
print(metrics.v_measure_score(y_train, y_pred_gak_km))
print(metrics.v_measure_score(y_train, y_pred_kshape))
# print(metrics.v_measure_score(y_train, yy))

print('\n-------------------v fowlkes_mallows_score------------------\n')

print(metrics.fowlkes_mallows_score(y_train, y_pred_km))
print(metrics.fowlkes_mallows_score(y_train, y_pred_gak_km))
print(metrics.fowlkes_mallows_score(y_train, y_pred_kshape))
# print(metrics.fowlkes_mallows_score(y_train, yy))




#%%
show_one_event(X_train, 1, 2)
show_one_event(X_train, 131, 2)
#%%
print('kmean sil value is: ', silhouette_score(per_unit, y_pred_km))
print('kernel k mean sil value is:', silhouette_score(per_unit, y_pred_gak_km))
print('kshape sil value is:', silhouette_score(per_unit, y_pred_kshape))
print('actual values are: ', silhouette_score(per_unit, y_train))
#%%
mlp = TimeSeriesMLPClassifier(hidden_layer_sizes=(64, 64), random_state=0)
mlp.fit(X_train, y_train)
mlp.score(X_train, y_train)
