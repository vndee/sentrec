import os
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, BertForSequenceClassification


class JointClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(JointClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("bert-base-cased")

        self.linear_1 = torch.nn.Linear(in_features=input_dim, out_features=128)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(in_features=128, out_features=num_classes)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, y, z):
        x = self.bert(input_ids=x, attention_mask=y, output_attentions=True, output_hidden_states=True).pooler_output
        x = torch.cat([x, z], dim=1)
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

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def __getitem__(self, item):
        node_feature = torch.cat([self.u[item], self.v[item]], dim=0)
        text_feature = self.tokenizer(self.t[item], truncation=True, padding="max_length", max_length=256)
        return torch.tensor(text_feature["input_ids"]), torch.tensor(text_feature["attention_mask"]), node_feature, \
               self.y[item]

    def __len__(self):
        return self.y.shape[0]


def compute_metrics(input, target):
    input = np.argmax(input, -1)
    acc = accuracy_score(target, input)
    f1 = f1_score(target, input, average="macro")
    return acc, f1


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="ArgumentParser")
    argument_parser.add_argument("-u_tr", "--u_train", type=str, default="data/mini1k/u_train.pt")
    argument_parser.add_argument("-v_tr", "--v_train", type=str, default="data/mini1k/v_train.pt")
    argument_parser.add_argument("-u_te", "--u_test", type=str, default="data/mini1k/u_test.pt")
    argument_parser.add_argument("-v_te", "--v_test", type=str, default="data/mini1k/v_test.pt")
    argument_parser.add_argument("-r_tr", "--r_train", type=str, default="data/mini1k/train.csv")
    argument_parser.add_argument("-r_te", "--r_test", type=str, default="data/mini1k/test.csv")
    argument_parser.add_argument("-d", "--device", type=str, default="cuda")
    argument_parser.add_argument("-e", "--epoch", type=int, default=100)
    argument_parser.add_argument("-l", "--learning_rate", type=float, default=1e-5)
    argument_parser.add_argument("-s", "--save_dir", type=str, default="data/checkpoint")
    argument_parser.add_argument("-p", "--pretrained", type=str, default=None)
    args = argument_parser.parse_args()

    u_train, v_train = torch.load(args.u_train), torch.load(args.v_train)
    u_test, v_test = torch.load(args.v_test), torch.load(args.v_test)
    df = pd.read_csv(args.r_train)
    y_train = torch.from_numpy(np.array(df.Score.astype(int).tolist()) - 1)
    text_train = df.Text.tolist()
    df = pd.read_csv(args.r_test)
    y_test = torch.from_numpy(np.array(df.Score.astype(int).tolist()) - 1)
    text_test = df.Text.tolist()

    #---
    train_loader = torch.utils.data.DataLoader(VectorDataset(u=u_train, v=v_train, t=text_train, y=y_train), shuffle=True, batch_size=4)
    test_loader = torch.utils.data.DataLoader(VectorDataset(u=u_test, v=v_test, t=text_test, y=y_test), shuffle=True, batch_size=4)

    with torch.no_grad():
        v = torch.zeros((text_train.__len__(), 768))
        bert = AutoModel.from_pretrained("bert-base-cased").to(args.device)
        for i, (x, y, z, t) in enumerate(tqdm(train_loader, desc="Processing train")):
            x, y = x.to(args.device), y.to(args.device)
            p = bert(input_ids=x, attention_mask=y).pooler_output
            p = p.detach().cpu()
            v[i * 4: i * 4 + 4] = p

        print(v.shape)
        torch.save("data/train_vec.pt", v)
    #---

    net = JointClassifier(input_dim=896, num_classes=5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=args.learning_rate, params=net.parameters())

    if args.pretrained is not None:
        net = net.load_state_dict(torch.load(args.pretrained, map_location=torch.device(args.device)))
        print(f"Found valid pretrained model and loaded from {args.pretrained}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(VectorDataset(u=u_train, v=v_train, t=text_train, y=y_train), shuffle=True, batch_size=4)
    test_loader = torch.utils.data.DataLoader(VectorDataset(u=u_test, v=v_test, t=text_test, y=y_test), shuffle=True, batch_size=4)

    best_perf = 0.
    net = net.to(args.device)
    for epoch in range(args.epoch):
        # train
        net.train()
        loss_tr, cnt, acc_tr, f1_tr = 0., 0, 0., 0.
        for x, y, z, t in tqdm(train_loader, desc=f"Training {epoch + 1}/{args.epoch}"):
            x, y, z, t = x.to(args.device), y.to(args.device), z.to(args.device), t.to(args.device)
            p = net(x, y, z)
            loss = criterion(p, t)
            loss.backward()
            optimizer.zero_grad()

            loss_tr = loss_tr + loss.item()
            cnt = cnt + 1

            if args.device == "cuda":
                p = p.detach().cpu().numpy()
                t = t.detach().cpu().numpy()
            else:
                p = p.detach().numpy()
                t = t.detach().numpy()

            acc, f1 = compute_metrics(p, t)
            acc_tr = acc_tr + acc
            f1_tr = f1_tr + f1

        loss_tr = loss_tr / cnt
        acc_tr = acc_tr / cnt
        f1_tr = f1_tr / cnt

        # test
        net.eval()
        loss_te, cnt, acc_te, f1_te = 0., 0, 0., 0.
        with torch.no_grad():
            for x, y, z, t in tqdm(test_loader, desc=f"Testing {epoch + 1}/{args.epoch}"):
                x, y, z, t = x.to(args.device), y.to(args.device), z.to(args.device), t.to(args.device)
                p = net(x, y, z)
                loss = criterion(p, t)
                loss_te = loss_te + loss.item()
                cnt = cnt + 1

                if args.device == "cuda":
                    p = p.detach().cpu().numpy()
                    t = t.detach().cpu().numpy()
                else:
                    p = p.detach().numpy()
                    t = t.detach().numpy()

                acc, f1 = compute_metrics(p, t)
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
