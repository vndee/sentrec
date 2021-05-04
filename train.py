import os
import torch
import random
import argparse
import numpy as np

from models import GCNNet, SAGE, RGCNNet, SEALNet
from loader import AmazonFineFoodsReviews
from torch_geometric.data import ClusterData, DataLoader
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
    argument.add_argument('-r', '--learning_rate', type=float, default=0.001, help='Model learning rate')
    argument.add_argument('-d', '--device', type=str, default='cpu', help='Training device')
    argument.add_argument('-e', '--epoch', type=int, default=100, help='The number of epoch')
    argument.add_argument('-t', '--text_feature', type=bool, default=False, help='Using text feature or not')
    argument.add_argument('-s', '--multi_task', type=bool, default=False, help='Using multi-task training')
    argument.add_argument('-m', '--max_length', type=int, default=512, help='Reviews max length')
    argument.add_argument('-n', '--num_partition', type=int, default=1, help='Number of graph partition')
    argument.add_argument('-k', '--num_hops', type=int, default=3, help='Number of hops')
    argument.add_argument('-c', '--model', type=str, default='seal', help='Model')
    argument.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    argument.add_argument('-a', '--random_seed', type=int, default=42, help='Seed number')
    args = argument.parse_args()
    set_reproducibility_state(args.random_seed)

    print(args)
    graph = AmazonFineFoodsReviews(database_path=args.input).build_graph(
        text_feature=args.text_feature,
        language_model_name=args.language_model_shortcut,
        max_length=args.max_length)

    if args.model in ['gcn', 'rgcn', 'sage']:
        cluster_data = ClusterData(graph, num_parts=args.num_partition, recursive=True)
        # cluster_data = NeighborSampler(edge_index=graph.edge_index, sizes=[-1])
        # cluster_data = GraphSAINTRandomWalkSampler(graph, walk_length=3, num_steps=100, batch_size=32)
        print('Graph partitioned..')

        if args.model == 'gcn':
            net = GCNNet()
        elif args.model == 'rgcn':
            net = RGCNNet()
        else:
            net = SAGE()

        net = net.to(args.device)
        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)

        best_val_perf, test_perf = 0., 0.
        for epoch in range(args.epoch):
            total_test_acc, total_test_f1 = 0., 0.
            cnt, total_train_loss, total_val_perf, total_temp_test_perf = 0, 0., 0., 0.
            total_train_acc, total_train_f1, total_val_acc, total_val_f1 = 0., 0., 0., 0.

            for cluster in cluster_data:
                if args.multi_task is False:
                    cluster.y = cluster.y + 1

                cluster = split_graph(cluster)
                cluster = to_undirected(cluster)

                cluster = cluster.to(args.device)
                train_loss, train_acc, train_f1 = net.learn(data=cluster, optimizer=optim, criterion=criterion,
                                                            device=args.device)
                val_perf, temp_test_perf = net.evaluate(data=cluster, device=args.device)

                cnt = cnt + 1
                val_acc, val_f1 = val_perf[0], val_perf[1]
                test_acc, test_f1 = temp_test_perf[0], temp_test_perf[1]

                total_train_loss = total_train_loss + train_loss
                total_train_acc = total_train_acc + train_acc
                total_train_f1 = total_train_f1 + train_f1

                total_val_acc = total_val_acc + val_acc
                total_val_f1 = total_val_f1 + val_f1

                total_test_acc = total_test_acc + test_acc
                total_test_f1 = total_test_f1 + test_f1

            avg_train_loss = total_train_loss / cnt
            avg_train_acc = total_train_acc / cnt
            avg_train_f1 = total_train_f1 / cnt

            avg_val_acc = total_val_acc / cnt
            avg_val_f1 = total_val_f1 / cnt

            avg_test_acc = total_test_acc / cnt
            avg_test_f1 = total_test_f1 / cnt

            print(f'Epoch: {epoch + 1:04d}/{args.epoch:04d}, train_loss: {avg_train_loss:.5f}, '
                  f'train_acc: {avg_train_acc:.2f}, train_f1: {avg_train_f1:.2f}, '
                  f'val_acc: {avg_val_acc:.2f}, val_f1: {avg_val_f1:.2f}, '
                  f'test_acc: {avg_test_acc:.2f}, test_f1: {avg_test_f1:.2f}')

    elif args.model == 'seal':
        graph = split_graph(graph)
        graph = to_undirected(graph)
        seal = SEALDataset(split_graph(graph), args.num_hops)
        train_loader, val_loader, test_loader = DataLoader(seal.train, batch_size=args.batch_size, shuffle=True), \
                                                DataLoader(seal.val, batch_size=args.batch_size, shuffle=True), \
                                                DataLoader(seal.test, batch_size=args.batch_size, shuffle=True)

        net = SEALNet(seal).to(args.device)
        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)

        for epoch in range(args.epoch):
            # train
            total_test_acc, total_test_f1 = 0., 0.
            cnt, total_train_loss, total_val_perf, total_temp_test_perf = 0, 0., 0., 0.
            total_train_acc, total_train_f1, total_val_acc, total_val_f1 = 0., 0., 0., 0.

            for train_data in train_loader:
                loss, f1, acc = net.learn(data=train_data, optimizer=optim, criterion=criterion, device=args.device)
                cnt = cnt + 1
                total_train_loss = total_train_loss + loss
                total_train_acc = total_train_acc + acc
                total_train_f1 = total_train_f1 + f1

            avg_train_loss = total_train_loss / cnt
            avg_train_acc = total_train_acc / cnt
            avg_train_f1 = total_train_f1 / cnt

            for val_data in val_loader:
                f1, acc = net.evaluate(data=val_data, device=args.device)
                cnt = cnt + 1
                total_val_f1 = total_val_f1 + f1
                total_val_acc = total_val_acc + acc

            avg_val_f1 = total_val_f1 / cnt
            avg_val_acc = total_val_acc / cnt

            for test_data in test_loader:
                f1, acc = net.evaluate(data=test_data, device=args.device)
                cnt = cnt + 1
                total_test_f1 = total_test_f1 + f1
                total_test_acc = total_test_acc + acc

            avg_test_f1 = total_test_f1 / cnt
            avg_test_acc = total_test_acc / cnt

            print(f'Epoch: {epoch + 1:04d}/{args.epoch:04d}, train_loss: {avg_train_loss:.5f}, '
                  f'train_acc: {avg_train_acc:.2f}, train_f1: {avg_train_f1:.2f}, '
                  f'val_acc: {avg_val_acc:.2f}, val_f1: {avg_val_f1:.2f}, '
                  f'test_acc: {avg_test_acc:.2f}, test_f1: {avg_test_f1:.2f}')
