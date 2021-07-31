import os
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class LinearClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear_1 = torch.nn.Linear(in_features=input_dim, out_features=128)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(in_features=128, out_features=num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.softmax(x)
        return x


class VectorDataset(torch.utils.data.Dataset):
    def __init__(self, u=None, v=None, t=None, y=None):
        self.u = u
        self.v = v
        self.t = t
        self.y = y

    def __getitem__(self, item):
        return self.t[item], self.y[item]

    def __len__(self):
        return self.y.shape[0]


def compute_metrics(input, target):
    input = np.argmax(input, -1)
    acc = accuracy_score(target, input)
    f1 = f1_score(target, input, average="macro")
    return acc, f1


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="ArgumentParser")
    argument_parser.add_argument("-u", "--u", type=str, default="data/weights_100k/u.pt")
    argument_parser.add_argument("-v", "--v", type=str, default="data/weights_100k/v.pt")
    argument_parser.add_argument("-t", "--t", type=str, default="data/mini/train.vec")
    argument_parser.add_argument("-r", "--r", type=str, default="data/mini/train.csv")
    argument_parser.add_argument("-d", "--device", type=str, default="cpu")
    argument_parser.add_argument("-e", "--epoch", type=int, default=100)
    argument_parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
    argument_parser.add_argument("-s", "--save_dir", type=str, default="data/checkpoint")
    argument_parser.add_argument("-p", "--pretrained", type=str, default=None)
    args = argument_parser.parse_args()

    with open(args.t, "rb") as stream:
        t = torch.from_numpy(pickle.loads(stream.read()))

    u = torch.load(args.u)
    v = torch.load(args.v)

    df = pd.read_csv(args.r)
    y = torch.from_numpy(np.array(df.Score.astype(int).tolist()) - 1)

    test_size = 0.1
    trs, tes = None, None
    for it in range(0, 5):
        pos = np.where(y == it)[0]
        test_pos = np.random.choice(pos, int(pos.shape[0] * test_size))
        test_mask = np.isin(pos, test_pos)
        train_mask = ~test_mask
        train_set = pos[train_mask]
        test_set = pos[test_mask]
        trs = np.atleast_1d(train_set) if trs is None else np.concatenate([trs, train_set])
        tes = np.atleast_1d(test_set) if tes is None else np.concatenate([tes, test_set])

    train_t, train_u, train_v, train_y = t[trs], u[trs], v[trs], y[trs]
    test_t, test_u, test_v, test_y = t[tes], u[tes], v[tes], y[tes]

    net = LinearClassifier(input_dim=768, num_classes=5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=args.learning_rate, params=net.parameters())

    if args.pretrained is not None:
        net = net.load_state_dict(torch.load(args.pretrained, map_location=torch.device(args.device)))
        print(f"Found valid pretrained model and loaded from {args.pretrained}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(VectorDataset(t=train_t, y=train_y), shuffle=True, batch_size=4)
    test_loader = torch.utils.data.DataLoader(VectorDataset(t=test_t, y=test_y), shuffle=True, batch_size=4)

    best_perf = 0.
    net = net.to(args.device)
    for epoch in range(args.epoch):
        # train
        net.train()
        loss_tr, cnt, acc_tr, f1_tr = 0., 0, 0., 0.
        for x, y in tqdm(train_loader, desc=f"Training {epoch + 1}/{args.epoch}"):
            x, y = x.to(args.device), y.to(args.device)
            z = net(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.zero_grad()

            loss_tr = loss_tr + loss.item()
            cnt = cnt + 1

            if args.device == "cuda":
                z = z.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
            else:
                z = z.detach().numpy()
                y = y.detach().numpy()

            acc, f1 = compute_metrics(z, y)
            acc_tr = acc_tr + acc
            f1_tr = f1_tr + f1

        loss_tr = loss_tr / cnt
        acc_tr = acc_tr / cnt
        f1_tr = f1_tr / cnt

        # test
        net.eval()
        loss_te, cnt, acc_te, f1_te = 0., 0, 0., 0.
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"Testing {epoch + 1}/{args.epoch}"):
                x, y = x.to(args.device), y.to(args.device)
                z = net(x)
                loss = criterion(z, y)
                loss_te = loss_te + loss.item()
                cnt = cnt + 1

                if args.device == "cuda":
                    z = z.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                else:
                    z = z.detach().numpy()
                    y = y.detach().numpy()

                acc, f1 = compute_metrics(z, y)
                acc_te = acc_te + acc
                f1_te = f1_te + f1

        loss_te = loss_te / cnt
        acc_te = acc_te / cnt
        f1_te = f1_te / cnt

        if acc_te > best_perf:
            best_perf = acc_te
            torch.save(net.state_dict(), os.path.join(args.save_dir, f"{acc_te}_net.pt"))
            print("Found improvement and saved best model!!!")

        print(
            f"[Epoch {epoch + 1}/{args.epoch}] loss_tr: {loss_tr} - acc_tr: {acc_tr} - f1_tr: {f1_tr} | loss_te: {loss_te} - acc_te: {acc_te} - f1_te: {f1_te}")
