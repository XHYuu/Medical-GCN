import os
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torchnet as tnt


def load_data(main_path, use_node_labels=True):
    Gs = []
    labels = []

    print(f"Loading data from {main_path}")

    for label, subfolder in enumerate(['NSI', 'SI']):  # 调整顺序，将 'nc' 放在前面
        nodes_features_folder = os.path.join(main_path, subfolder, 'nodes_features')
        adj_matrix_folder = os.path.join(main_path, subfolder, 'adj_matrix')

        nodes_files = os.listdir(nodes_features_folder)

        print(f"Processing {subfolder}, assigned label {label}")

        for file_name in nodes_files:
            nodes_features_path = os.path.join(nodes_features_folder, file_name)
            adj_matrix_path = os.path.join(adj_matrix_folder, file_name)

            node_features_df = pd.read_csv(nodes_features_path, header=None)
            node_features = node_features_df.values
            num_brain_nodes = 8
            num_gut_nodes = 50
            assert node_features.shape[1] == num_brain_nodes + num_gut_nodes
            assert node_features.shape[0] == 20

            adj_matrix_df = pd.read_csv(adj_matrix_path, header=None)
            adj_matrix = adj_matrix_df.values
            assert adj_matrix.shape == (num_brain_nodes, num_gut_nodes)

            G = nx.Graph()

            for i in range(num_brain_nodes):
                node_label = node_features[:, i]
                G.add_node(f"brain_{i}", attr_dict=node_label)

            for j in range(num_gut_nodes):
                node_label = node_features[:, num_brain_nodes + j]
                G.add_node(f"gut_{j}", attr_dict=node_label)

            for i in range(num_brain_nodes):
                for j in range(num_gut_nodes):
                    if adj_matrix[i, j] != 0:
                        G.add_edge(f"brain_{i}", f"gut_{j}", weight=adj_matrix[i, j])

            Gs.append(G)
            labels.append(label)

    return list(zip(Gs, labels))


def get_graph_signal(nx_graph):
    x = []
    for _, attr in nx_graph.nodes(data=True):
        x.append(attr['attr_dict'])
    return np.array(x)


def to_pytorch_dataset(dataset, label_offset=0, batch_size=16):
    list_set = []
    for graph, label in dataset:
        F, G = get_graph_signal(graph), nx.to_numpy_array(graph)
        F_tensor = torch.from_numpy(F).float()
        G_tensor = torch.from_numpy(G).float()

        label += label_offset

        list_set.append((F_tensor, G_tensor, label))

    dataset_tnt = tnt.dataset.ListDataset(list_set)
    data_loader = torch.utils.data.DataLoader(dataset_tnt, shuffle=True, batch_size=batch_size)
    return data_loader


def create_loaders(dataset, train_index, val_index, batch_size, offset=0):
    train_dataset = [dataset[i] for i in train_index]
    val_dataset = [dataset[i] for i in val_index]
    return to_pytorch_dataset(train_dataset, offset, batch_size), to_pytorch_dataset(val_dataset, offset, batch_size)
