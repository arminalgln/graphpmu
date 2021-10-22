import torch
import pickle
import numpy as np
from models.graph_model import GraphPMU, GraphEncoder, GraphEncoderLocGlob, GraphPMULocalGlobal
from models.graph_model import GraphEncoderLocGlobAutoEncoder, EncGraphDisc


from models.discriminator import Discriminator
from models.AED.simpleAED import Encoder
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


with open('data/positive_graphs_timeseries.pkl', 'rb') as handle:
  pos_graphs = pickle.load(handle)

with open('data/negative_graphs_timeseries.pkl', 'rb') as handle:
  neg_graphs = pickle.load(handle)

#initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# g_encoder = GraphEncoder
encoder = Encoder
# g_encoder = GraphEncoderLocGlob
g_encoder = GraphEncoderLocGlobAutoEncoder
disc = Discriminator
node_nums = pos_graphs[0].num_nodes()
[seq_len, n_time_features, enc_emb_size] = [pos_graphs[0].ndata['features'].shape[0],
                                              pos_graphs[0].ndata['features'].shape[-1], 64]
[in_feats, h1_feats, last_space_feature] = [enc_emb_size, 128, 64]
[D_h1, D_h2] = [32, 32]
measure1 = 'JSD'#['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1', 'JSMI']
measure2 = 'BCE'#['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1', 'JSMI']
# graphpmu = GraphPMU(g_encoder, disc, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2, device)
# graphpmu = EncGraphDisc(g_encoder, disc, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2, device)
graphpmu = EncGraphDisc(encoder, g_encoder, disc, seq_len, n_time_features,enc_emb_size, node_nums, h1_feats,
                 last_space_feature, D_h1, D_h2, device)

#make positive and negative batches
num_samples = len(pos_graphs)

num_train = int(num_samples * 0.9)
np.random.seed(0)
train_selector = np.random.choice(num_samples, num_train, replace=False)
test_selector = np.setdiff1d(np.arange(num_samples), train_selector)

train_index = np.random.choice(train_selector.shape[0], 1000, replace=False)
test_index = np.random.choice(test_selector.shape[0], 100, replace=False)

train_selector = train_selector[train_index]
test_selector = test_selector[test_index]


train_sampler = SubsetRandomSampler(torch.from_numpy(train_selector))
test_sampler = SubsetRandomSampler(torch.from_numpy(test_selector))

b_size = 5

pos_train_dataloader = GraphDataLoader(
    pos_graphs, sampler=train_sampler, batch_size=b_size, drop_last=False)
pos_test_dataloader = GraphDataLoader(
    pos_graphs, sampler=test_sampler, batch_size=b_size, drop_last=False)
neg_train_dataloader = GraphDataLoader(
    neg_graphs, sampler=train_sampler, batch_size=b_size, drop_last=False)
neg_test_dataloader = GraphDataLoader(
    neg_graphs, sampler=test_sampler, batch_size=b_size, drop_last=False)
#
sample_graph = next(iter(pos_test_dataloader)).to(device)
sample_graph.ndata['embd'] = graphpmu.encoder(sample_graph.ndata['features'].to(device))
sample_graph.ndata['embd'], sample_graph.ndata['hcat'], H = graphpmu(sample_graph)
#%%
def train_graphpmu_loc_glob(graphpmu, pos_train_dataloader, neg_train_dataloader,
                   pos_test_dataloader, neg_test_dataloader, epochs_num, b_size, lr):
    #initialization for training
    graphpmu_optimizer = torch.optim.Adam(graphpmu.parameters(), lr=lr)#, weight_decay=1e-4
    # criteria = global_loss_
    criteria = local_global_loss_
    # BCE = nn.BCELoss()
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(graphpmu.state_dict())
    best_loss = 10000.0

    alpha = 1

    labels = torch.zeros(((2*b_size)**2)*node_nums, device=device)
    posidx = []

    # labels = torch.zeros((2*(b_size)**2)*node_nums, device=device)
    for i in range(2*b_size):
        labels[i*((2*b_size+1)*node_nums):node_nums+i*((2*b_size+1)*node_nums)] = 1
        posidx.append(np.arange(i * ((2 * b_size + 1) * node_nums), node_nums + i * ((2 * b_size + 1) * node_nums)))
    posidx = np.array(posidx).ravel()
    lblindx = np.arange(0, labels.shape[0])
    negidx = np.setdiff1d(lblindx, posidx)
    posidx = torch.tensor(posidx)
    negidx = torch.tensor(negidx)

    for epoch in range(epochs_num):
        #training mode
        graphpmu = graphpmu.train()
        graphpmu_optimizer.zero_grad()
        train_losses = []

        pos_iter = iter(pos_train_dataloader)
        neg_iter = iter(neg_train_dataloader)
        for pos_batch in pos_iter:#iter on train batches
            pos_batch.ndata['features'] = torch.ones([125, 125, 9])
            ng = next(neg_iter)
            ng.ndata['features'] = torch.ones([125, 125, 9])/2
            pos_neg_graphs = dgl.batch([pos_batch, ng]) #concat pos and neg graphs
            # pos_neg_graphs = dgl.batch([pos_batch, next(neg_iter)]) #concat pos and neg graphs
            # pos_neg_graphs.ndata['features'] = torch.ones([250, 125, 9])
            pos_neg_graphs = pos_neg_graphs.to(device)
            pos_neg_graphs.ndata['embd'], pos_neg_graphs.ndata['hcat'], H = graphpmu(pos_neg_graphs)
            hcat = pos_neg_graphs.ndata['hcat']
            Hsize = H.shape
            hcatsize = hcat.shape

            hcat = hcat.repeat(Hsize[0], 1)
            H = torch.unsqueeze(H, dim=1)
            H = H.expand(Hsize[0], hcatsize[0], Hsize[-1]).reshape(Hsize[0] * hcatsize[0], Hsize[-1])
            latent = torch.cat((H, hcat), axis=1)
            # latent = latent[0:int(H.shape[0] / 2)]

            pred = graphpmu.discriminator(latent)

            loss1 = criteria(pred, posidx, negidx, measure1)
            # loss2 = criteria(pred, labels, measure2)
            # loss = alpha * loss1 + (1-alpha) * loss2
            loss = alpha * loss1
            # loss = (1-alpha) * loss2
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
                pos_neg_graphs = pos_neg_graphs.to(device)
                pos_neg_graphs.ndata['embd'], pos_neg_graphs.ndata['hcat'], H = graphpmu(pos_neg_graphs)
                hcat = pos_neg_graphs.ndata['hcat']
                Hsize = H.shape
                hcatsize = hcat.shape

                hcat = hcat.repeat(Hsize[0], 1)
                H = torch.unsqueeze(H, dim=1)
                H = H.expand(Hsize[0], hcatsize[0], Hsize[-1]).reshape(Hsize[0] * hcatsize[0], Hsize[-1])

                latent = torch.cat((H, hcat), axis=1)
                # latent = latent[0:int(H.shape[0] / 2)]

                pred = graphpmu.discriminator(latent)

                loss1 = criteria(pred, posidx, negidx, measure1)
                # loss2 = criteria(pred, labels, measure2)
                # loss = alpha * loss1 + (1 - alpha) * loss2
                loss = alpha * loss1
                # loss = (1 - alpha) * loss2
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
    return graphpmu.eval(), train_loss

lr = 1e-3
gpmodel, train_loss = train_graphpmu_loc_glob(graphpmu, pos_train_dataloader, neg_train_dataloader,
                                              pos_test_dataloader, neg_test_dataloader, epochs_num=5, b_size=b_size,
                                              lr=lr)
# gpmodel = train_graphpmu_loc_glob(graphpmu, pos_train_dataloader, neg_train_dataloader,
#                          pos_test_dataloader, neg_test_dataloader, epochs_num=10, b_size=b_size)
#%%
err = 10
lr = 1e-3
sigma = 0.01
mu = 0.001
batch_size = [20, 10, 5, 2]
learning_rates = [0.01, 0.001]
for b_size in batch_size:
    for lr in learning_rates:
        print(b_size, lr)
        # lr = np.abs(sigma * np.random.randn() + mu)
        # b_size = 5
        pos_train_dataloader = GraphDataLoader(
            pos_graphs, sampler=train_sampler, batch_size=b_size, drop_last=False)
        pos_test_dataloader = GraphDataLoader(
            pos_graphs, sampler=test_sampler, batch_size=b_size, drop_last=False)
        neg_train_dataloader = GraphDataLoader(
            neg_graphs, sampler=train_sampler, batch_size=b_size, drop_last=False)
        neg_test_dataloader = GraphDataLoader(
            neg_graphs, sampler=test_sampler, batch_size=b_size, drop_last=False)
        # initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # g_encoder = GraphEncoder
        encoder = Encoder
        # g_encoder = GraphEncoderLocGlob
        g_encoder = GraphEncoderLocGlobAutoEncoder
        disc = Discriminator
        node_nums = pos_graphs[0].num_nodes()
        [seq_len, n_time_features, enc_emb_size] = [pos_graphs[0].ndata['features'].shape[0],
                                                    pos_graphs[0].ndata['features'].shape[-1], 32]
        [in_feats, h1_feats, last_space_feature] = [enc_emb_size, 128, 64]
        [D_h1, D_h2] = [1, 1]
        measure2 = 'JSD'  # ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1', 'JSMI']
        measure1 = 'BCE'  # ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1', 'JSMI']
        print(measure1)
        # graphpmu = GraphPMU(g_encoder, disc, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2, device)
        # graphpmu = EncGraphDisc(g_encoder, disc, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2, device)
        graphpmu = EncGraphDisc(encoder, g_encoder, disc, seq_len, n_time_features, enc_emb_size, node_nums, h1_feats,
                                last_space_feature, D_h1, D_h2, device)

        # make positive and negative batches
        num_samples = len(pos_graphs)
        print(num_samples)
        gpmodel, train_loss = train_graphpmu_loc_glob(graphpmu, pos_train_dataloader, neg_train_dataloader,
                                          pos_test_dataloader, neg_test_dataloader, epochs_num=1, b_size=b_size, lr=lr)
        err = train_loss
        print(train_loss)
        if err < 0:
            break
#%%
gpmodel, train_loss = train_graphpmu_loc_glob(gpmodel, pos_train_dataloader, neg_train_dataloader,
                                      pos_test_dataloader, neg_test_dataloader, epochs_num=100, b_size=b_size, lr=lr)
#%%
torch.save(gpmodel, 'models/saved/locglob_withconcat_pos_neg_ones_bsize10_lr-3')
#%%
# graphpmu = GraphPMULocalGlobal(g_encoder, disc, node_nums, in_feats, h1_feats, last_space_feature, D_h1, D_h2,
#                                device)
gpmodel = torch.load('models/saved/locglob_withconcat_pos_neg_ones_bsize10_lr-3')
gpmodel.eval()
#%%
from sklearn.metrics import accuracy_score
#discriminator evaluation
def get_accuracy(y_true, y_prob):
    cls = []
    count = 0
    for c, i in enumerate(y_prob):
        if i > 0.5:
            cls.append(1)
        else:
            cls.append(0)
        if cls[c] == y_true[c]:
            count += 1
    return count/len(cls)
with torch.no_grad():
    pos_iter = iter(pos_test_dataloader)
    neg_iter = iter(neg_test_dataloader)
    count = 0

    labels = torch.zeros(((2*b_size)**2)*node_nums, device=device)
    for i in range(b_size):
        labels[i*((2*b_size+1)*node_nums):node_nums+i*((2*b_size+1)*node_nums)] = 1


    # labels = torch.zeros(((2 * b_size) ** 2) * node_nums, device=device)
    # for i in range(2 * b_size):
    #     labels[i * ((2 * b_size + 1) * node_nums):node_nums + i * ((2 * b_size + 1) * node_nums)] = 1

    for pos_batch in pos_iter:  # iter on train batches
        # print(count)
        count += 1
        pos_neg_graphs = dgl.batch([pos_batch, next(neg_iter)])  # concat pos and neg graphs
        # g_enc = gpmodel.encoder(pos_neg_graphs)

        pos_neg_graphs = pos_neg_graphs.to(device)
        pos_neg_graphs.ndata['embd'], pos_neg_graphs.ndata['hcat'], H = graphpmu(pos_neg_graphs)
        hcat = pos_neg_graphs.ndata['hcat']
        Hsize = H.shape
        hcatsize = hcat.shape

        hcat = hcat.repeat(Hsize[0], 1)
        H = torch.unsqueeze(H, dim=1)
        H = H.expand(Hsize[0], hcatsize[0], Hsize[-1]).reshape(Hsize[0] * hcatsize[0], Hsize[-1])

        latent = torch.cat((H, hcat), axis=1)
        # latent = latent[0:int(H.shape[0] / 2)]

        pred = graphpmu.discriminator(latent)
        print(pred.shape, labels.shape)

        # pred = gpmodel.discriminator(g_enc)
        # pred = gpmodel(pos_neg_graphs)
        # y = torch.cat((torch.ones(pos_batch.batch_size), torch.zeros(pos_batch.batch_size)))
        # target = torch.cat(
        #     (torch.ones(pos_batch.batch_size, device=device), torch.zeros(pos_batch.batch_size, device=device)))
        # loss = BCE(pred, target)
        print(get_accuracy(labels, pred))

