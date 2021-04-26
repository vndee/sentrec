import os
import torch
import random
import argparse
import numpy as np

from gcn import GCNNet
from utils import split_graph
from loader import AmazonFineFoodsReviews
from torch_geometric.data import ClusterData, ClusterLoader


def set_reproducibility(seed):
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
                          help='Pre-trained language model shortcut')
    argument.add_argument('-r', '--learning_rate', type=float, default=0.01, help='Model learning rate')
    argument.add_argument('-d', '--device', type=str, default='cpu', help='Training device')
    argument.add_argument('-e', '--epoch', type=int, default=100, help='The number of epoch')
    argument.add_argument('-t', '--text_feature', type=bool, default=False, help='Using text feature or not')
    argument.add_argument('-s', '--multi_task', type=bool, default=False, help='Using multi-task training')
    argument.add_argument('-m', '--max_length', type=int, default=512, help='Reviews max length')
    argument.add_argument('-a', '--random_seed', type=int, default=42, help='Seed number')
    args = argument.parse_args()

    set_reproducibility(args.random_seed)
    net = GCNNet()
    graph = AmazonFineFoodsReviews(database_path=args.input).build_graph(
        text_feature=args.text_feature,
        language_model_name=args.language_model_shortcut,
        max_length=args.max_length)

    graph = ClusterData(graph, num_parts=3)
    cluster_graph = ClusterLoader(graph, batch_size=32, shuffle=True)

    total_num_nodes = 0
    for step, sub_data in enumerate(cluster_graph):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of nodes in the current batch: {sub_data.num_nodes}')
        print(sub_data)
        print()
        total_num_nodes += sub_data.num_nodes

    print(f'Iterated over {total_num_nodes} of {graph.num_nodes} nodes!')
    # graph = split_graph(graph)
    # # verify_negative_edge(graph)
    #
    # if args.multi_task is False:
    #     graph.y = graph.y + 1
    #
    # net, graph = net.to(args.device), graph.to(args.device)
    # criterion = torch.nn.CrossEntropyLoss()
    # optim = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)
    #
    # best_val_perf, test_perf = 0., 0.
    # for epoch in range(args.epoch):
    #     train_loss = net.learn(data=graph, optimizer=optim, criterion=criterion, device=args.device)
    #     val_perf, temp_test_perf = net.evaluate(data=graph, device=args.device)
    #
    #     if val_perf > best_val_perf:
    #         best_val_perf = val_perf
    #         test_perf = temp_test_perf
    #
    #     print(f'Epoch: {epoch:04d}/{args.epoch:04d}, '
    #           f'Loss: {train_loss:.5f}, '
    #           f'Val: {best_val_perf:.5f}, '
    #           f'Test: {test_perf:.5f}')
