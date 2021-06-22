import torch
from torch_geometric.nn import RGCNConv, GCNConv, SAGEConv
from transformers import AutoModel
from sklearn.metrics import f1_score, accuracy_score


class GCNJointRepresentation(torch.nn.Module):
    def __init__(self, num_classes=5, language_model='bert-base-cased', hidden_dim=768, conv_type='RGCN'):
        super(GCNJointRepresentation, self).__init__()

        # GCN
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

        # Join representation
        self.linear1 = torch.nn.Linear(hidden_dim + 64, 128)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, data, edge_index, edge_map):
        z = self.encode(data)

        edge_attr = torch.zeros(edge_index.shape[1], edge_map.get_dim())
        for it in range(edge_index.shape[1]):
            u, v = edge_index[0][it].item(), edge_index[1][it].item()
            edge_attr[it] = edge_map.get(u, v)

        return self.decode(z, edge_index, edge_attr)

    def encode(self, data):
        if self.conv_type == 'rgcn':
            x = self.conv1(data.x, data.train_edge_index, torch.ones(data.train_edge_index.shape[1])).relu()
            x = self.conv2(x, data.train_edge_index, torch.ones(data.train_edge_index.shape[1])).relu()
        else:
            x = self.conv1(data.x, data.train_edge_index).relu()
            x = self.conv2(x, data.train_edge_index).relu()

        return x

    def decode(self, z, edge_index, edge_attr):
        node_representation = (z[edge_index[0]] * z[edge_index[1]])
        joint_representation = torch.cat([node_representation, edge_attr], dim=-1)
        x = self.linear1(joint_representation)
        x = self.relu(x)
        x = self.linear2(x)
        return self.softmax(x)

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

    def learn(self, data, optimizer: torch.optim.Optimizer, device: torch.device, edge_map, criterion):
        self.train()
        optimizer.zero_grad()

        z = self.encode(data)

        edge_attr = torch.zeros(data.train_edge_index.shape[1], edge_map.get_dim())
        for it in range(data.train_edge_index.shape[1]):
            u, v = data.train_edge_index[0][it].item(), data.train_edge_index[1][it].item()
            edge_attr[it] = torch.from_numpy(edge_map.get(u, v)).float()

        link_logits = self.decode(z, data.train_edge_index, edge_attr.to(device))
        loss = criterion(link_logits, data.train_target_index.to(device))
        loss.backward()
        optimizer.step()

        link_preds = torch.argmax(link_logits, dim=-1).cpu().detach().numpy()
        data.train_target_index = data.train_target_index.cpu().detach().numpy()

        return loss.item(), accuracy_score(data.train_target_index, link_preds), f1_score(
            data.train_target_index, link_preds, average='macro')

    @torch.no_grad()
    def evaluate(self, data, edge_map, criterion, device: torch.device):
        self.eval()

        tgt_edge_index = data[f'val_target_index']
        pos_edge_index = data[f'val_edge_index']

        edge_attr = torch.zeros(pos_edge_index.shape[1], edge_map.get_dim())
        for it in range(pos_edge_index.shape[1]):
            u, v = pos_edge_index[0][it].item(), pos_edge_index[1][it].item()
            edge_attr[it] = torch.from_numpy(edge_map.get(u, v)).float()

        z = self.encode(data)
        link_logits = self.decode(z, pos_edge_index, edge_attr.to(device))
        loss = criterion(link_logits, tgt_edge_index.to(device))

        link_preds = torch.argmax(link_logits, dim=-1).cpu().detach().numpy()
        tgt_edge_index = tgt_edge_index.cpu().detach().numpy()

        acc = accuracy_score(tgt_edge_index, link_preds)
        f1 = f1_score(tgt_edge_index, link_preds, average='macro')

        return loss, acc, f1
