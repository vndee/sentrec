import os
import time
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from loader import AmazonFineFoodsReviews, chunks
from transformers import AdamW, get_scheduler, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from models import GCNJointRepresentation, SEALJointRepresentation
from torch_geometric.data import ClusterData, DataLoader
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
    argument.add_argument('-i', '--input', type=str, default='data/mini/train', help='Path to training data')
    argument.add_argument('-y', '--test', type=str, default='data/mini/test', help='Path to testing file')
    argument.add_argument('-l', '--language_model_shortcut', type=str, default='bert-base-cased',
                          help='Pre-trained language models shortcut')
    argument.add_argument('-r', '--learning_rate', type=float, default=1e-4, help='Model learning rate')
    argument.add_argument('-d', '--device', type=str, default='cpu', help='Training device')
    argument.add_argument('-e', '--epoch', type=int, default=1000, help='The number of epoch')
    argument.add_argument('-t', '--text_feature', type=bool, default=True, help='Using text feature or not')
    argument.add_argument('-s', '--multi_task', type=bool, default=False, help='Using multi-task training')
    argument.add_argument('-m', '--max_length', type=int, default=512, help='Reviews max length')
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
    graph, text = AmazonFineFoodsReviews(database_path=args.input, test_path=args.test).build_graph(
        text_feature=args.text_feature)

    if args.text_feature:
        t0, input_ids, attention_mask = time.time(), None, None
        tokenizer = AutoTokenizer.from_pretrained(args.language_model_shortcut)
        for it, te in enumerate(chunks(text, args.batch_size) * 8):
            tokenized = tokenizer(te, padding="max_length", truncation=True, max_length=args.max_length,
                                  return_tensors="np")
            inp, attn = tokenized["input_ids"], tokenized["attention_mask"]

            input_ids = np.atleast_1d(inp) if input_ids is None else np.concatenate([input_ids, inp])
            attention_mask = np.atleast_1d(attn) if attention_mask is None else np.concatenate([attention_mask, attn])

            els = time.time() - t0
            est = (els / (it + 1)) * ((len(text) / args.batch_size) - it - 1)
            print(f"Tokenized elapsed {timedelta(seconds=els)} - estimate {timedelta(seconds=est)}")

        print(input_ids.shape)
        print(attention_mask.shape)

    with open(f"{args.input}.vec", "rb") as stream:
        edge_attr = pickle.loads(stream.read())

    with open(f"{args.test}.vec", "rb") as stream:
        test_edge_attr = pickle.loads(stream.read())

    pivot = edge_attr.shape[0]
    edge_attr = np.concatenate([edge_attr, test_edge_attr])

    print(f"Graph: {graph.edge_index.shape}")
    print(f"Edge: {edge_attr.shape}")

    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))

    if args.model in ['gcn', 'rgcn', 'sage']:
        edge_map = EdgeHashMap(graph.edge_index, edge_attr)
        del edge_attr

        cluster_data = ClusterData(graph, num_parts=args.num_partition, recursive=True)
        print('Graph partitioned..')

        net = GCNJointRepresentation(conv_type=args.model)
        if args.pretrained is not None:
            net.load_state_dict(torch.load(args.pretrained))
            print(f"Loaded pretrained weights from {args.pretrained}")

        net = net.to(args.device)

        criterion = torch.nn.CrossEntropyLoss()
        optim = AdamW(net.parameters(), lr=args.learning_rate)

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optim,
            num_warmup_steps=0,
            num_training_steps=args.epoch
        )

        best_perf, t0 = 0., time.time()
        for epoch in range(args.epoch):
            total_test_acc, total_test_f1 = 0., 0.
            cnt, total_train_loss, total_val_perf, total_temp_test_perf = 0, 0., 0., 0.
            total_train_acc, total_train_f1, total_val_acc, total_val_f1 = 0., 0., 0., 0.

            total_val_loss, total_test_loss = 0., 0.

            for cluster in tqdm(cluster_data, f"Training {1 + epoch}/{args.epoch}"):
                cluster = split_graph(cluster, pivot=pivot)
                cluster = to_undirected(cluster)
                cluster = cluster.to(args.device)

                train_loss, train_acc, train_f1 = net.learn(data=cluster, scheduler=lr_scheduler, optimizer=optim,
                                                            criterion=criterion, edge_map=edge_map, device=args.device)
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
            writer.add_scalar('learning_rate', lr_scheduler.get_last_lr()[0], epoch)

            els = time.time() - t0
            est = (els / (1 + epoch)) * (args.epoch - epoch - 1)
            print(f"Time elapsed: {timedelta(seconds=els)} - Time estimate: {timedelta(seconds=est)}")

    elif args.model == 'seal':
        graph.edge_attr = edge_attr[:1000]
        split_graph(graph)
        graph = to_undirected(graph)
        seal = SEALDataset(graph, args.num_hops)
        train_loader, val_loader = DataLoader(seal.train, batch_size=args.batch_size, shuffle=True), DataLoader(
            seal.val, batch_size=args.batch_size, shuffle=True)

        net = SEALJointRepresentation(seal).to(args.device)
        criterion = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)

        for epoch in range(args.epoch):
            # train
            total_test_acc, total_test_f1 = 0., 0.
            cnt, total_train_loss, total_val_perf, total_temp_test_perf = 0, 0., 0., 0.
            total_train_acc, total_train_f1, total_val_acc, total_val_f1 = 0., 0., 0., 0.

            for train_data in tqdm(train_loader, desc=f'Training {1 + epoch}/{args.epoch}'):
                loss, f1, acc = net.learn(data=train_data, optimizer=optim, criterion=criterion,
                                          device=args.device)
                cnt = cnt + 1
                total_train_loss = total_train_loss + loss
                total_train_acc = total_train_acc + acc
                total_train_f1 = total_train_f1 + f1

            avg_train_loss = total_train_loss / cnt
            avg_train_acc = total_train_acc / cnt
            avg_train_f1 = total_train_f1 / cnt

            avg_val_f1, avg_val_acc = batch_evaluate(net, val_loader, args.device)

            print(f'Epoch: {epoch + 1:04d}/{args.epoch:04d}, train_loss: {avg_train_loss:.5f}, '
                  f'train_acc: {avg_train_acc:.2f}, train_f1: {avg_train_f1:.2f}, '
                  f'val_acc: {avg_val_acc:.2f}, val_f1: {avg_val_f1:.2f}, ')

    writer.close()
