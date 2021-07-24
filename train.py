import os
import time
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from loader import AmazonFineFoodsReviews, chunks, TokenizedDataset
from transformers import AdamW, get_scheduler, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from models import GCNJointRepresentation, SEALJointRepresentation
from torch_geometric.data import DataLoader, ClusterData
from utils import SEALDataset, split_graph, to_undirected, EdgeHashMap


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
    argument.add_argument('-i', '--input', type=str, default='data/Reviews', help='Path to training data')
    argument.add_argument('-y', '--test', type=str, default='data/mini/test', help='Path to testing file')
    argument.add_argument('-l', '--language_model_shortcut', type=str, default='bert-base-cased',
                          help='Pre-trained language models shortcut')
    argument.add_argument('-r', '--learning_rate', type=float, default=1e-5, help='Model learning rate')
    argument.add_argument('-d', '--device', type=str, default='cuda', help='Training device')
    argument.add_argument('-e', '--epoch', type=int, default=10000, help='The number of epoch')
    argument.add_argument('-t', '--text_feature', type=bool, default=False, help='Using text feature or not')
    argument.add_argument('-s', '--multi_task', type=bool, default=False, help='Using multi-task training')
    argument.add_argument('-m', '--max_length', type=int, default=256, help='Reviews max length')
    argument.add_argument('-n', '--num_partition', type=int, default=1, help='Number of graph partition')
    argument.add_argument('-k', '--num_hops', type=int, default=3, help='Number of hops')
    argument.add_argument('-c', '--model', type=str, default='sage', help='Model')
    argument.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    argument.add_argument('-a', '--random_seed', type=int, default=42, help='Seed number')
    argument.add_argument('-g', '--save_dir', type=str, default='data/weights/', help='Path to save dir')
    argument.add_argument('-p', '--pretrained', type=str, default='data/weights/best.pt',
                          help='Path to pretrained model')
    args = argument.parse_args()
    set_reproducibility_state(args.random_seed)

    print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    graph = AmazonFineFoodsReviews(database_path=args.input).build_graph()

    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))

    if args.model in ['gcn', 'rgcn', 'sage']:
        net = GCNJointRepresentation(conv_type=args.model)
        if args.pretrained is not None:
            net.load_state_dict(torch.load(args.pretrained), strict=False)
            print(f"Loaded pretrained weights from {args.pretrained}")

        net = net.to(args.device)

        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        best_perf, t0 = 0., time.time()
        for epoch in range(args.epoch):
            print(f"Training {1 + epoch}/{args.epoch}..")
            graph = to_undirected(graph)
            graph = graph.to(args.device)
            train_loss, train_acc, train_f1 = net.learn(data=graph, optimizer=optim,
                                                        criterion=criterion, device=args.device)

            print(f'Epoch: {epoch + 1:04d}/{args.epoch:04d}, train_loss: {train_loss:.5f}, '
                  f'train_acc: {train_acc:.2f}, train_f1: {train_f1:.2f}')

            if train_acc > best_perf:
                with torch.no_grad():
                    best_perf = train_acc
                    torch.save(net.state_dict(), os.path.join(args.save_dir, 'best.pt'))
                    net.eval()
                    z = net.encode(graph)
                    u, v = z[graph.train_edge_index[0]].cpu(), z[graph.train_edge_index[1]].cpu()
                    torch.save(u, os.path.join(args.save_dir, "u.pt"))
                    torch.save(v, os.path.join(args.save_dir, "v.pt"))
                    print("Saved best model..")

            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_f1', train_f1, epoch)

            els = time.time() - t0
            est = (els / (1 + epoch)) * (args.epoch - epoch - 1)
            print(f"Time elapsed: {timedelta(seconds=els)} - Time estimate: {timedelta(seconds=est)}")
