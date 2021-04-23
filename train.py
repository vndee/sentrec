import torch
import argparse

from gcn import GCNNet
from utils import train_test_split_edges
from loader import AmazonFineFoodsReviews

if __name__ == '__main__':
    argument = argparse.ArgumentParser(description='Training job for Sentiment Graph for Recommendation')
    argument.add_argument('-i', '--input', type=str, default='data/Reviews.csv', help='Path to training data')
    argument.add_argument('-l', '--language_model_shortcut', type=str, default='bert-base-cased',
                          help='Pre-trained language model shortcut')
    argument.add_argument('-r', '--learning_rate', type=float, default=0.01, help='Model learning rate')
    argument.add_argument('-d', '--device', type=str, default='cpu', help='Training device')
    argument.add_argument('-e', '--epoch', type=int, default=50, help='The number of epoch')
    argument.add_argument('-t', '--text_feature', type=bool, default=True, help='Using text feature or not')
    argument.add_argument('-s', '--multi_task', type=bool, default=False, help='Using multi-task training')
    argument.add_argument('-m', '--max_length', type=int, default=512, help='Reviews max length')
    args = argument.parse_args()

    net = GCNNet()
    graph = AmazonFineFoodsReviews(database_path=args.input).build_graph(
        text_feature=args.text_feature,
        language_model_name=args.language_model_shortcut,
        max_length=args.max_length)
    graph = train_test_split_edges(graph)

    net, graph = net.to(args.device), graph.to(args.device)
    optim = torch.optim.Adam(params=net.parameters(), lr=args.learning_rate)

    best_val_perf, test_perf = 0., 0.
    for epoch in range(args.epoch):
        train_loss = net.learn(data=graph, optimizer=optim, device=args.device)
        val_perf, temp_test_perf = net.evaluate(data=graph, device=args.device)

        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = temp_test_perf

        print(f'Epoch: {epoch:04d}/{args.epoch:04d}, '
              f'Loss: {train_loss:.5f}, '
              f'Val: {best_val_perf:.5f}, '
              f'Test: {test_perf:.5f}')
