import os
import torch
import argparse

from gcn import GCNNet
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
from loader import AmazonFineFoodsReviews
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges


def get_link_labels(pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor, device: torch.device):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[: pos_edge_index.size(1)] = 1.
    return link_labels


def train(model: GCNNet, data: Data, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    negative_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    optimizer.zero_grad()
    z = model.encode(data)
    link_logits = model.decode(z, data.train_pos_edge_index, negative_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, negative_edge_index, device)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model: GCNNet, data: Data, device: torch.device):
    model.eval()
    perfs = []
    for prfx in ['val', 'test']:
        pos_edge_index = data[f'{prfx}_pos_edge_index']
        neg_edge_index = data[f'{prfx}_neg_edge_index']

        z = model.encode(data)
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index, device)
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))

    return perfs


if __name__ == '__main__':
    argument = argparse.ArgumentParser(description='Training job for Sentiment Graph for Recommendation')
    argument.add_argument('-i', '--input', type=str, default='data/Reviews.csv', help='Path to training data')
    argument.add_argument('-l', '--language_model_shortcut', type=str, default='bert-base-cased',
                          help='Pre-trained language model shortcut')
    argument.add_argument('-r', '--learning_rate', type=float, default=0.01, help='Model learning rate')
    argument.add_argument('-d', '--device', type=str, default='cpu', help='Training device')
    argument.add_argument('-e', '--epoch', type=int, default=50, help='The number of epoch')
    args = argument.parse_args()

    net = GCNNet()
    graph = AmazonFineFoodsReviews(database_path=args.input).build_graph(
        language_model_name=args.language_model_shortcut)
    graph = train_test_split_edges(graph)

    net, graph = net.to(args.device), graph.to(args.device)
    optim = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)

    best_val_perf, test_perf = 0., 0.
    for epoch in range(args.epoch):
        train_loss = train(model=net, data=graph, optimizer=optim, device=args.device)
        val_perf, temp_test_perf = evaluate(model=net, data=graph, device=args.device)

        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = temp_test_perf

        print(f'Epoch: {epoch:04d}/{args.epoch:04d}, '
              f'Loss: {train_loss:.5f}, '
              f'Val: {best_val_perf:.5f}, '
              f'Test: {test_perf:.5f}')
