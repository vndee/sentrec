import torch
from torch_geometric.nn import RGCNConv
from transformers import AutoModel
from sklearn.metrics import f1_score, accuracy_score


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

        link_preds = torch.argmax(link_logits, dim=-1)
        return loss.item(), accuracy_score(link_labels.cpu(), link_preds.cpu()), f1_score(link_labels.cpu(),
                                                                                          link_preds.cpu(),
                                                                                          average='macro')

    @torch.no_grad()
    def evaluate(self, data, device: torch.device):
        self.eval()
        acc, f1 = [], []
        for prfx in ['val', 'test']:
            tgt_edge_index = data[f'{prfx}_target_index']
            pos_edge_index = data[f'{prfx}_edge_index']
            neg_edge_index = data[f'{prfx}_neg_index']

            z = self.encode(data)
            link_logits = self.decode(z, pos_edge_index, neg_edge_index)
            link_preds = torch.argmax(link_logits, dim=-1)
            link_labels = RGCNNet.get_link_labels(neg_edge_index, tgt_edge_index, device)
            acc.append(accuracy_score(link_labels.cpu(), link_preds.cpu()))
            f1.append(f1_score(link_labels.cpu(), link_preds.cpu(), average='macro'))

        return acc, f1


class RGCNJointRepresentation(torch.nn.Module):
    def __init__(self, num_classes=5, language_model='bert-base-cased', hidden_dim=768):
        super(RGCNJointRepresentation, self).__init__()

        # GCN
        self.conv1 = RGCNConv(in_channels=1, out_channels=128, num_relations=1)
        self.conv2 = RGCNConv(in_channels=128, out_channels=64, num_relations=1)

        # LM
        self.lm = AutoModel.from_pretrained(language_model)

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
        x = self.conv1(data.x, data.train_edge_index, torch.ones(data.train_edge_index.shape[1])).relu()
        x = self.conv2(x, data.train_edge_index, torch.ones(data.train_edge_index.shape[1])).relu()
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

        link_preds = torch.argmax(link_logits, dim=-1)
        return loss.item(), accuracy_score(data.train_target_index, link_preds.cpu().numpy()), f1_score(
            data.train_target_index, link_preds.cpu().numpy(), average='macro')

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

        link_preds = torch.argmax(link_logits, dim=-1)
        acc = accuracy_score(tgt_edge_index, link_preds.cpu().numpy())
        f1 = f1_score(tgt_edge_index, link_preds.cpu().numpy(), average='macro')

        return loss, acc, f1
