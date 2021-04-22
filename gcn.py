import torch
from torch_geometric.nn import GCNConv


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