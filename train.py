import os
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from loader import AmazonFineFoodsReviews
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import NeighborSampler
from models import GCNJointRepresentation
from torch_geometric.data import ClusterData, DataLoader
from torch_geometric.data import GraphSAINTRandomWalkSampler
from utils import SEALDataset, split_graph, verify_negative_edge, to_undirected, EdgeHashMap


def set_reproducibility_state(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def batch_evaluate(net, loader, device):
    total_f1, total_acc = 0, 0.
    for data in loader:
        f1, acc = net.evaluate(data=data, device=device)
        total_f1 = total_f1 + f1
        total_acc = total_acc + acc

    return total_f1 / len(loader), total_acc / len(loader)


if __name__ == '__main__':
    argument = argparse.ArgumentParser(description='Training job for Sentiment Graph for Recommendation')
    argument.add_argument('-i', '--input', type=str, default='data/mini/train.csv', help='Path to training data')
    argument.add_argument('-f', '--edge_attr_file', type=str, default='data/mini/train.vec', help='Path to edge attribute index')
    argument.add_argument('-l', '--language_model_shortcut', type=str, default='bert-base-cased',
                          help='Pre-trained language models shortcut')
    argument.add_argument('-r', '--learning_rate', type=float, default=1e-4, help='Model learning rate')
    argument.add_argument('-d', '--device', type=str, default='cpu', help='Training device')
    argument.add_argument('-e', '--epoch', type=int, default=10, help='The number of epoch')
    argument.add_argument('-t', '--text_feature', type=bool, default=True, help='Using text feature or not')
    argument.add_argument('-s', '--multi_task', type=bool, default=False, help='Using multi-task training')
    argument.add_argument('-m', '--max_length', type=int, default=512, help='Reviews max length')
    argument.add_argument('-n', '--num_partition', type=int, default=1, help='Number of graph partition')
    argument.add_argument('-k', '--num_hops', type=int, default=3, help='Number of hops')
    argument.add_argument('-c', '--model', type=str, default='rgcn', help='Model')
    argument.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    argument.add_argument('-a', '--random_seed', type=int, default=42, help='Seed number')
    argument.add_argument('-g', '--save_dir', type=str, default='data/weights/', help='Path to save dir')
    args = argument.parse_args()
    set_reproducibility_state(args.random_seed)

    print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    graph = AmazonFineFoodsReviews(database_path=args.input).build_graph()

    with open(args.edge_attr_file, "rb") as stream:
        edge_attr = pickle.loads(stream.read())

    edge_map = EdgeHashMap(graph.edge_index, edge_attr)
    del edge_attr

    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))

    if args.model in ['gcn', 'rgcn', 'sage']:
        cluster_data = ClusterData(graph, num_parts=args.num_partition, recursive=True)
        # cluster_data = NeighborSampler(edge_index=graph.edge_index, sizes=[-1])
        # cluster_data = GraphSAINTRandomWalkSampler(graph, walk_length=3, num_steps=100, batch_size=32)
        print('Graph partitioned..')

        net = GCNJointRepresentation(conv_type=args.model)
        net = net.to(args.device)

        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)

        best_perf = 0.
        for epoch in range(args.epoch):
            total_test_acc, total_test_f1 = 0., 0.
            cnt, total_train_loss, total_val_perf, total_temp_test_perf = 0, 0., 0., 0.
            total_train_acc, total_train_f1, total_val_acc, total_val_f1 = 0., 0., 0., 0.

            total_val_loss, total_test_loss = 0., 0.

            for cluster in tqdm(cluster_data, f"Training {1 + epoch}/{args.epoch}"):
                cluster = split_graph(cluster)
                cluster = to_undirected(cluster)
                cluster = cluster.to(args.device)

                train_loss, train_acc, train_f1 = net.learn(data=cluster, optimizer=optim, criterion=criterion,
                                                            edge_map=edge_map, device=args.device)
                val_loss, val_acc, val_f1 = net.evaluate(data=cluster, criterion=criterion, edge_map=edge_map,
                                                         device=args.device)

                cnt = cnt + 1

                total_train_loss = total_train_loss + train_loss
                total_train_acc = total_train_acc + train_acc
                total_train_f1 = total_train_f1 + train_f1

                total_val_loss = total_val_loss + val_loss
                total_val_acc = total_val_acc + val_acc
                total_val_f1 = total_val_f1 + val_f1

            avg_train_loss = total_train_loss / cnt
            avg_train_acc = total_train_acc / cnt
            avg_train_f1 = total_train_f1 / cnt

            avg_val_loss = total_val_loss / cnt
            avg_val_acc = total_val_acc / cnt
            avg_val_f1 = total_val_f1 / cnt

            print(f'Epoch: {epoch + 1:04d}/{args.epoch:04d}, train_loss: {avg_train_loss:.5f}, '
                  f'train_acc: {avg_train_acc:.2f}, train_f1: {avg_train_f1:.2f}, '
                  f'val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.2f}, val_f1: {avg_val_f1:.2f}')

            if avg_train_acc > best_perf:
                best_perf = avg_train_acc
                torch.save(net.state_dict(), os.path.join(args.save_dir, 'best.pt'))

            writer.add_scalar('train_acc', avg_train_acc, epoch)
            writer.add_scalar('train_loss', avg_train_loss, epoch)
            writer.add_scalar('train_f1', avg_train_f1, epoch)
            writer.add_scalar('val_acc', avg_val_acc, epoch)
            writer.add_scalar('val_loss', avg_val_loss, epoch)
            writer.add_scalar('val_f1', avg_val_f1, epoch)

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

            avg_val_f1, avg_val_acc = batch_evaluate(net, val_loader, args.device)
            avg_test_f1, avg_test_acc = batch_evaluate(net, test_loader, args.device)

            print(f'Epoch: {epoch + 1:04d}/{args.epoch:04d}, train_loss: {avg_train_loss:.5f}, '
                  f'train_acc: {avg_train_acc:.2f}, train_f1: {avg_train_f1:.2f}, '
                  f'val_acc: {avg_val_acc:.2f}, val_f1: {avg_val_f1:.2f}, '
                  f'test_acc: {avg_test_acc:.2f}, test_f1: {avg_test_f1:.2f}')

    writer.close()
