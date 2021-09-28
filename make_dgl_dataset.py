import dgl
from dgl.data import DGLDataset
import torch
import os
import numpy as np
import pandas as pd
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

# features = np.load('data/features.npy')
# labels = np.load('data/labels.npy')
# data = np.load('data/all_event_data.npy')
# per_unit = np.load('data/all_per_unit.npy')


class PmuDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='PMU')

    def process(self):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print('Using device:', device)
        per_unit = np.load('data/aug_all_per_unit_806_824_836_846.npy')
        labels = np.load('data/aug_labels_806_824_836_846.npy')
        self.graphs = []
        self.labels = []
        self.info = {}

        # Create a graph for each event from per unit data and labels.

        labels = pd.DataFrame({'labels': labels})
        labels = labels['labels'].astype('category').cat.codes.to_numpy()
        self.labels = torch.LongTensor(labels)

        for event_data in per_unit:
            src = torch.from_numpy(np.array([0, 1, 1]))
            dst = torch.from_numpy(np.array([1, 2, 3]))
            num_nodes = 4
            node_features = torch.from_numpy(np.array(np.split(event_data, num_nodes, axis=1)))
            node_features = node_features.permute(0, 2, 1)
            node_features = torch.flatten(node_features, start_dim=1)
            # node_features = node_features.reshape(num_nodes, -1) #flastten the feature (for now)

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            g.ndata['features'] = node_features
            g = dgl.add_self_loop(g)
            # print(g.device)
            cuda_g = g  # accepts any device objects from backend framework
            # print(cuda_g.device)
            self.graphs.append(cuda_g)

        # self.info['num_nodes'] = num_nodes
        # self.info['num_classes'] = np.unique(labels).shape[0]
        # self.info['num_edges'] = src.shape[0]
        # self.info['num_features'] = node_features.shape[-1]
    # def save(self, graph_path):
    #     # save graphs and labels
    #     # graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    #     save_graphs(graph_path, self.graphs, {'labels': self.labels})
    #     # save other information in python dict
    #     # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    #     # save_info(info_path, {'num_classes': self.num_classes})
    #
    # def load(self, graph_path):
    #     # load processed data from directory `self.save_path`
    #     # graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    #     self.graphs, label_dict = load_graphs(graph_path)
    #     self.labels = label_dict['labels']
    #     info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    #     self.num_classes = load_info(info_path)['num_classes']

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


# dataset = PmuDataset()
# # # # graph_path = 'data/datasets/pmu_dgl_graph.bin'
# # # # dataset.save()
# graph, label = dataset[0]
# print(graph, label)
# print(graph.ndata['features'].shape)
