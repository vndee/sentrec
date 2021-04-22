import torch
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling


class GCNNet(torch.nn.Module):
    def __init__(self):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_channels=1, out_channels=128)
        self.conv2 = GCNConv(in_channels=128, out_channels=64)

    def forward(self, data, negative_edge_index):
        z = self.encode(data)
        return self.decode(z, data.train_pos_edge_index, neg_edge_index=negative_edge_index)

    def encode(self, data):
        x = self.conv1(data.x, data.train_pos_edge_index)
        x = x.relu()
        return self.conv2(x, data.train_pos_edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    @staticmethod
    def get_link_labels(pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor, device: torch.device):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=device)
        link_labels[: pos_edge_index.size(1)] = 1.
        return link_labels

    def learn(self, data, optimizer: torch.optim.Optimizer, device: torch.device):
        self.train()
        negative_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))

        optimizer.zero_grad()
        z = self.encode(data)
        link_logits = self.decode(z, data.train_pos_edge_index, negative_edge_index)
        link_labels = GCNNet.get_link_labels(data.train_pos_edge_index, negative_edge_index, device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, device: torch.device):
        self.eval()
        perfs = []
        for prfx in ['val', 'test']:
            pos_edge_index = data[f'{prfx}_pos_edge_index']
            neg_edge_index = data[f'{prfx}_neg_edge_index']

            z = self.encode(data)
            link_logits = self.decode(z, pos_edge_index, neg_edge_index)
            link_probs = link_logits.sigmoid()
            link_labels = GCNNet.get_link_labels(pos_edge_index, neg_edge_index, device)
            perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))

        return perfs
