import math
import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.nn import GCNConv, global_sort_pool
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Softmax


class SEALNet(torch.nn.Module):
    def __init__(self, dataset, hidden_channels=32, num_layers=3, num_classes=4, GNN=GCNConv, k=0.6):
        super(SEALNet, self).__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in dataset.train])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.convs = ModuleList()
        self.convs.append(GNN(dataset.num_features, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, num_classes)
        self.softmax = Softmax(dim=-1)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return self.softmax(x)

    def learn(self, data, optimizer, criterion, device):
        self.train()
        optimizer.zero_grad()

        truth = data.y.to(device).long()
        logits = self.forward(data.x, data.edge_index, data.batch)
        loss = criterion(logits, truth)
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, -1)
        f1, acc = f1_score(truth, pred), accuracy_score(truth, pred)

        return loss.item(), f1, acc

    def evaluate(self, data, device):
        self.eval()
        truth = data.y.to(device).long()
        logits = self.forward(data.x, data.edge_index, data.batch)

        pred = torch.argmax(logits, -1)
        f1, acc = f1_score(truth, pred), accuracy_score(truth, pred)
        return f1, acc
