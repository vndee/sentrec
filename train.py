import os
import torch
import random
import argparse
import numpy as np

from models import GCNNet, SAGE, RGCNNet
from loader import AmazonFineFoodsReviews
from torch_geometric.data import ClusterData
from torch_geometric.data import NeighborSampler
from torch_geometric.data import GraphSAINTRandomWalkSampler
from utils import SEALDataset, split_graph, verify_negative_edge, to_undirected


def set_reproducibility_state(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    argument = argparse.ArgumentParser(description='Training job for Sentiment Graph for Recommendation')
    argument.add_argument('-i', '--input', type=str, default='data/Reviews.csv', help='Path to training data')
    argument.add_argument('-l', '--language_model_shortcut', type=str, default='bert-base-cased',
                          help='Pre-trained language models shortcut')
    argument.add_argument('-r', '--learning_rate', type=float, default=0.01, help='Model learning rate')
    argument.add_argument('-d', '--device', type=str, default='cpu', help='Training device')
    argument.add_argument('-e', '--epoch', type=int, default=100, help='The number of epoch')
    argument.add_argument('-t', '--text_feature', type=bool, default=False, help='Using text feature or not')
    argument.add_argument('-s', '--multi_task', type=bool, default=False, help='Using multi-task training')
    argument.add_argument('-m', '--max_length', type=int, default=512, help='Reviews max length')
    argument.add_argument('-n', '--num_partition', type=int, default=1, help='Number of graph partition')
    argument.add_argument('-a', '--random_seed', type=int, default=42, help='Seed number')
    args = argument.parse_args()
    set_reproducibility_state(args.random_seed)

    net = RGCNNet()
    graph = AmazonFineFoodsReviews(database_path=args.input).build_graph(
        text_feature=args.text_feature,
        language_model_name=args.language_model_shortcut,
        max_length=args.max_length)

    graph = split_graph(graph)
    graph = to_undirected(graph)
    seal = SEALDataset(split_graph(graph), 3)

    # cluster_data = ClusterData(graph, num_parts=args.num_partition, recursive=True)
    # # cluster_data = NeighborSampler(edge_index=graph.edge_index, sizes=[-1])
    # # cluster_data = GraphSAINTRandomWalkSampler(graph, walk_length=3, num_steps=100, batch_size=32)
    # print('Graph partitioned..')
    #
    # net = net.to(args.device)
    # criterion = torch.nn.CrossEntropyLoss()
    # optim = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)
    #
    # best_val_perf, test_perf = 0., 0.
    # for epoch in range(args.epoch):
    #     cnt, total_train_loss, total_val_perf, total_temp_test_perf = 0, 0., 0., 0.
    #
    #     for cluster in cluster_data:
    #         cluster = split_graph(cluster)
    #
    #         if args.multi_task is False:
    #             cluster.y = cluster.y + 1
    #
    #         cluster = cluster.to(args.device)
    #         train_loss = net.learn(data=cluster, optimizer=optim, criterion=criterion, device=args.device)
    #         val_perf, temp_test_perf = net.evaluate(data=cluster, device=args.device)
    #
    #         cnt = cnt + 1
    #         total_train_loss = total_train_loss + train_loss
    #         total_val_perf = total_val_perf + val_perf
    #         total_temp_test_perf = total_temp_test_perf + temp_test_perf
    #
    #     train_loss = total_train_loss / cnt
    #     val_perf = total_val_perf / cnt
    #     test_perf = total_temp_test_perf / cnt
    #
    #     print(f'Epoch: {epoch + 1:04d}/{args.epoch:04d}, '
    #           f'Loss: {train_loss:.5f}, '
    #           f'Val: {val_perf:.5f}, '
    #           f'Test: {test_perf:.5f}')
