import torch
from torch_geometric.nn import RGCNConv
from sklearn.metrics import roc_auc_score, accuracy_score


class RGCNNet(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(RGCNNet, self).__init__()
        self.conv1 = RGCNConv(in_channels=1, out_channels=128, num_relations=1)
        self.conv2 = RGCNConv(in_channels=128, out_channels=64, num_relations=1)
        self.linear = torch.nn.Linear(64, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, data, negative_edge_index):
        z = self.encode(data)
        return self.decode(z, data.train_pos_edge_index, neg_edge_index=negative_edge_index)

    def encode(self, data):
        x = self.conv1(data.x, data.train_edge_index, torch.ones(data.train_edge_index.shape[1])).relu()
        x = self.conv2(x, data.train_edge_index, torch.ones(data.train_edge_index.shape[1])).relu()
        return self.linear(x)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]])
        return self.softmax(logits)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

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
        link_logits = self.decode(z, data.train_edge_index, data.train_neg_index)
        link_labels = RGCNNet.get_link_labels(data.train_neg_index, data.train_target_index, device)
        loss = criterion(link_logits, link_labels)
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, device: torch.device):
        self.eval()
        perfs = []
        for prfx in ['val', 'test']:
            tgt_edge_index = data[f'{prfx}_target_index']
            pos_edge_index = data[f'{prfx}_edge_index']
            neg_edge_index = data[f'{prfx}_neg_index']

            z = self.encode(data)
            link_logits = self.decode(z, pos_edge_index, neg_edge_index)
            link_preds = torch.argmax(link_logits, dim=-1)
            link_labels = RGCNNet.get_link_labels(neg_edge_index, tgt_edge_index, device)
            perfs.append(accuracy_score(link_labels.cpu(), link_preds.cpu()))

        return perfs
