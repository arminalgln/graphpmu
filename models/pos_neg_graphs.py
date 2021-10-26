import numpy as np
import torch
import dgl
import pickle
import pandas as pd
import networkx as nx
#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #load the data
# per_unit = np.load('data/whole_data_ss.pkl', allow_pickle=True)
# labels = np.load('data/new_aug_labels_806_824_836_846.npy')
# bus_data = pd.read_excel('data/ss.xlsx')
network = pd.read_excel('data/edges.xlsx')
# node_data = np.load('data/whole_buses_latent.pkl', allow_pickle=True)
# node_data = np.load('data/whole_data_ss.pkl', allow_pickle=True)
node_data = np.load('data/latent_with_just_pmu_AED.pkl', allow_pickle=True)
src = network['src']
dst = network['dst']
#%%

def make_positive_graphs(node_data, src, dst, device):
    # d = device
    pmus = list(node_data.keys())
    # sample_nums, _, _ = node_data[pmus[0]].shape # three unpack for time series
    sample_nums, _ = node_data[pmus[0]].shape # two unpack for latent
    print(sample_nums)
    pos_graphs = []
    for graph in range(sample_nums):
        # print(graph)
        g = dgl.graph((src, dst), num_nodes=len(pmus))
        g = dgl.add_self_loop(g)
        # g = g.to(device)
        node_features = []
        for pmu in pmus:
            node_features.append(node_data[pmu][graph])
        # print(pmu)
        # node_features = torch.from_numpy(np.array(node_features)).to(device)
        node_features = torch.from_numpy(np.array(node_features))
        g.ndata['features'] = node_features
        pos_graphs.append(g)
    return pos_graphs

pos_graphs = make_positive_graphs(node_data, src, dst, device)
#%%
#save the positive graphs
# with open('data/positive_graphs.pkl', 'wb') as handle:
#     pickle.dump(pos_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

#save the positive graphs
with open('data/positive_graphs_latent_with_just_pmu_AED.pkl', 'wb') as handle:
    pickle.dump(pos_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

#save the positive graphs with time series data
# with open('data/positive_graphs_timeseries.pkl', 'wb') as handle:
#     pickle.dump(pos_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('data/positive_graphs.pkl', 'rb') as handle:
#     pos_graphs = pickle.load(handle)
#%%
#making negative graphs (trees)

# tree = nx.random_tree(n=25, seed=2)
# print(nx.forest_str(tree, sources=[0]))
# print(nx.to_dict_of_lists(tree))
def make_negative_graphs(positive_graphs, device):
    neg_graphs = []
    for i, graph in enumerate(positive_graphs):
        # print(graph)
        node_nums = graph.num_nodes()
        rnd_tree = nx.random_tree(n=node_nums, seed=i)
        edges = list(rnd_tree.edges)
        src = [edges[i][0] for i in range(len(edges))]
        dst = [edges[i][1] for i in range(len(edges))]
        g = dgl.graph((src, dst), num_nodes=node_nums)
        g = dgl.add_self_loop(g)
        # g = g.to(device)
        g.ndata['features'] = graph.ndata['features']
        neg_graphs.append(g)
    return neg_graphs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neg_graphs = make_negative_graphs(pos_graphs, device)

#%%
#save the negative graphs
# with open('data/negative_graphs.pkl', 'wb') as handle:
#     pickle.dump(neg_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/negative_graphs_latent_with_just_pmu_AED.pkl', 'wb') as handle:
    pickle.dump(neg_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)


#save the negative graphs with time series data
# with open('data/negative_graphs_timeseries.pkl', 'wb') as handle:
#     pickle.dump(neg_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)