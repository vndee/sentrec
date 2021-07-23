import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import RGCNConv, GCNConv, SAGEConv
from sklearn.metrics import f1_score, accuracy_score


class GCNJointRepresentation(torch.nn.Module):
    def __init__(self, num_classes=5, conv_type='RGCN'):
        super(GCNJointRepresentation, self).__init__()

        if conv_type == 'rgcn':
            self.conv1 = RGCNConv(in_channels=1, out_channels=128, num_relations=1)
            self.conv2 = RGCNConv(in_channels=128, out_channels=64, num_relations=1)
        elif conv_type == 'gcn':
            self.conv1 = GCNConv(in_channels=1, out_channels=128)
            self.conv2 = GCNConv(in_channels=128, out_channels=64)
        elif conv_type == 'sage':
            self.conv1 = SAGEConv(in_channels=1, out_channels=128)
            self.conv2 = SAGEConv(in_channels=128, out_channels=64)

        self.conv_type = conv_type

        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(64, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def encode(self, data):
        if self.conv_type == 'rgcn':
            x = self.conv1(data.x, data.edge_index, torch.ones(data.edge_index.shape[1])).relu()
            x = self.conv2(x, data.edge_index, torch.ones(data.edge_index.shape[1])).relu()
        else:
            x = self.conv1(data.x, data.edge_index).relu()
            x = self.conv2(x, data.edge_index).relu()

        return x

    def decode(self, z, edge_index):
        node_representation = (z[edge_index[0]] * z[edge_index[1]])
        node_representation = self.linear(node_representation)
        return self.softmax(node_representation)

    @staticmethod
    def get_link_labels(neg_edge_index: torch.Tensor,
                        target_value: torch.Tensor,
                        device: torch.device):
        link_labels = torch.cat([target_value, torch.zeros(neg_edge_index.size(1), dtype=torch.long, device=device)],
                                dim=-1)
        return link_labels.long()

    def learn(self, data, optimizer: torch.optim.Optimizer, device: torch.device, criterion):
        self.train()
        optimizer.zero_grad()

        z = self.encode(data)
        link_logits = self.decode(z, data.train_edge_index)
        link_labels = data.train_y
        loss = criterion(link_logits, link_labels)
        loss.backward()
        optimizer.step()

        links = torch.argmax(link_logits, dim=-1)

        if device == "cuda":
            links = links.detach().cpu().numpy()
            link_labels = link_labels.detach().cpu().numpy()

        return loss.item(), accuracy_score(link_labels, links), f1_score(link_labels, links, average="macro")
