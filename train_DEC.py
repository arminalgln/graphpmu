# from sequitur.models import LSTM_AE
import torch
from torch import nn
import numpy as np
# from sequitur import quick_train
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from models.AED.simpleAED import Encoder, Decoder, RecurrentAutoencoder, LatentMu
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn, optim
import pandas as pd
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch.autograd import Variable
import scipy
#%%
# mat = scipy.io.loadmat('file.mat')
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


#define and train a model and save
model = RecurrentAutoencoder(125, 36, 32)
model.to(device)
model.float()
#TODO
def update_mu(train_dataloader, enc, device):
    sampler = iter(train_dataloader)
    samples_for_clusters = []
    for i in range(15):
        z = enc(sampler.next().to(device))
        samples_for_clusters.append(z)
    samples_for_clusters = torch.cat(samples_for_clusters, dim=0)
    samples_for_clusters = samples_for_clusters.detach().cpu()
    kmeans = KMeans(9, n_init=20)
    kmeans.fit(samples_for_clusters)
    mu = torch.from_numpy(kmeans.cluster_centers_).to(device)
    mu = Variable(mu, requires_grad=True).to(device)
    return mu
#%%
def train_model(model, train_dataset, val_dataset, n_epochs):
    enc = model.encoder
    dec = model.decoder

    criterion = nn.MSELoss(reduction='mean')
    enc_criteria = LatentMu.apply
    mu_learning_rate = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    enc_optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-5)
    dec_optimizer = torch.optim.Adam(dec.parameters(), lr=1e-3, weight_decay=1e-5)

    dec_criterion = nn.MSELoss(reduction='mean')
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    #initial mu for 1000 events latent
    mu = update_mu(train_dataloader, enc, device)

    for epoch in range(1, n_epochs + 1):
        model = model.train()
        enc = model.encoder
        enc = enc.train()

        dec = model.decoder

        train_losses = []
        train_losses_enc = []
        for batch_input in train_dataset:
            batch_input = batch_input.to(device)

            # train AED with reconstruction loss
            predicted = model(batch_input)
            optimizer.zero_grad()

            loss = criterion(predicted.float(), batch_input.float())

            loss.backward()
            optimizer.step()
            # train encoder with latent space aux pdf
            #each 5 times
            if epoch % 5 == 0:
                enc_optimizer.zero_grad()
                latent = enc(batch_input)

                loss_enc = enc_criteria(latent, mu)
                loss_enc.backward()

                enc_optimizer.step()
            if epoch % 10 == 0:
                mu.data -= mu_learning_rate * mu.grad.data


            train_losses.append(loss.item())
            # train_losses_enc.append(loss_enc.item())

        # update mu
        # if epoch % 10 == 0:
        #     mu = update_mu(train_dataloader, enc, device)


        val_losses = []
        val_losses_enc = []
        model = model.eval()
        enc = enc.eval()
        with torch.no_grad():
            for batch_input in val_dataset:
                batch_input = batch_input.to(device)
                pred = model(batch_input)
                loss = criterion(pred.float(), batch_input.float())

                latent = enc(batch_input)
                loss_enc = enc_criteria(latent, mu)

                val_losses.append(loss.item())
                val_losses_enc.append(loss_enc.item())

        train_loss = np.mean(train_losses)
        # train_loss_enc = np.mean(train_losses_enc)
        val_loss = np.mean(val_losses)
        val_loss_enc = np.mean(val_losses_enc)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            model_path = 'models/AED/806_824_836_846_stacked_DEC'
            torch.save(model, model_path)
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}  val loss enc {val_loss_enc}')
    model.load_state_dict(best_model_wts)
    return model.eval(), history

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
for ev in [102, 483]:
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
model_path = 'models/AED/806_824_836_846_stacked_DEC'
model = torch.load(model_path)
model.eval()
#%%
labels = pd.DataFrame({'labels': labels})
labels = labels['labels'].astype('category').cat.codes.to_numpy()

all_data = torch.from_numpy(per_unit).to(device)
# get the latent variables

selected_latent = model.encoder(all_data[train_selector[2387:3580]]).cpu().detach().numpy()
selected_labels = labels[train_selector[2387:3580]]
selected_latent = model.encoder(all_data[test_selector]).cpu().detach().numpy()
selected_labels = labels[test_selector]


#%%
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
all_clustering_models(selected_latent, selected_labels, cluster_num)

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(selected_latent)

plt.scatter(X_embedded[:,0], X_embedded[:,1],c=selected_labels)
plt.legend(selected_labels)
plt.show()
#%%
model = RecurrentAutoencoder(125, 36, 32)
model.to(device)
model.float()
sample = next(iter(train_dataloader)).to(device)

print(next(iter(model.parameters()))[0])
y = model(sample)
enc = model.encoder
z = enc(sample)
dec = model.decoder
xhat = dec(z)
optmizer = optim.Adam(model.parameters())
criterion = criterion = nn.MSELoss(reduction='mean')
optmizer.zero_grad()
loss = criterion(xhat.float(), sample.float())
# print(next(iter(model.parameters())))
loss.backward()
optmizer.step()
print(next(iter(model.parameters()))[0])
#%%
enc = model.encoder
z = enc(sample)
dec = model.decoder
xhat = dec(z)

print(next(iter(enc.parameters()))[0][0])
print(next(iter(dec.parameters()))[0][0])

encoder_opt = optim.Adam(enc.parameters())
decoder_opt = optim.Adam(dec.parameters())

criterion = criterion = nn.MSELoss(reduction='mean')
decoder_opt.zero_grad()
loss = criterion(xhat.float(), sample.float())
# print(next(iter(model.parameters())))
loss.backward()
decoder_opt.step()

print(next(iter(enc.parameters()))[0][0])
print(next(iter(dec.parameters()))[0][0])

#%%
model_path = 'models/AED/806_824_836_846_stacked'
model = torch.load(model_path)
model.eval()

#%%
def update_mu(train_dataloader, enc, device):
    sampler = iter(train_dataloader)
    samples_for_clusters = []
    for i in range(15):
        z = enc(sampler.next().to(device))
        samples_for_clusters.append(z)
    samples_for_clusters = torch.cat(samples_for_clusters, dim=0)
    samples_for_clusters = samples_for_clusters.detach().cpu()
    kmeans = KMeans(9, n_init=20)
    kmeans.fit(samples_for_clusters)
    mu = torch.from_numpy(kmeans.cluster_centers_).to(device)
    mu = Variable(mu, requires_grad=True).to(device)
    return mu
#%%
model = RecurrentAutoencoder(125, 36, 32)
model.to(device)
model.float()
sample = next(iter(train_dataloader)).to(device)
enc = model.encoder
#%%
lm = LatentMu.apply
from sklearn.cluster import KMeans

mu = update_mu(train_dataloader, enc, device)
sample_at = 1000
k = int(np.floor(sample_at/10))
#%%
# sample = next(itertools.islice(train_dataloader, 1000, None)).to(device)
z=enc(sample)


lm = LatentMu.apply


mu_learning_rate = 1e-4
print(mu[0])
# print(mu.grad)
# z = enc(batch_input)
l = lm(z, mu)
# print(next(iter(enc.parameters()))[0][0])
encoder_opt = optim.Adam(enc.parameters())
encoder_opt.zero_grad()
l.backward()
encoder_opt.step()
# print(next(iter(enc.parameters()))[0][0])
# print(mu.grad)
mu.data -= mu_learning_rate * mu.grad.data
print(mu[0])
# a=EmbeddingCluster(1, 9)
# a.initial_fit(z)
# q=a.q_measure(z)
# p=a.aux_dist(z)
# ll = a.loss_fun(z)
# gg = a.grad(z)
#%%
def q_measure(z, mu):
    p = 1.0
    dist = torch.cdist(z, mu, 2).float()
    q = 1.0 / (1 + dist ** 2 )
    q = (q.T / q.sum(axis=1)).T
    return q


def aux_dist(z, mu):
    z = z
    q = q_measure(z, mu)
    q = (q.T / q.sum(axis=1)).T
    p = (q ** 2)
    p = (p.T / p.sum(axis=1)).T
    return p
z=z.float().to(device)
mu = mu.float().to(device)
q = q_measure(z,mu)
p = aux_dist(z, mu)
d = 1 + torch.cdist(z, mu, 2)**2
constant_val = (p-q)/d

constant_val_T = constant_val.T

extented_z = z.repeat((mu.shape[0], 1))
extented_mu = torch.repeat_interleave(mu, z.shape[0], dim=0)
simple_dist_z_mu = (extented_z - extented_mu).reshape(9,100,32)

constant_val_T_rep = constant_val_T.repeat_interleave(32, axis=1)
constant_val_T_rep = constant_val_T_rep.reshape(9,100,32)


all_grads = 2 * constant_val_T_rep * simple_dist_z_mu
# ml = ml/di

#test
i, j = 97, 2
print(2*(p[i,j]-q[i,j])*(z[i]-mu[j])/(d[i,j]))
print(all_grads[j,i])

