import os
import torch
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, \
    BertForSequenceClassification


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


class CNNClassifier(torch.nn.Module):
    def __init__(self, num_classes=5):
        super(CNNClassifier, self).__init__()
        self.conv_1 = torch.nn.Conv1d(in_channels=9, out_channels=3, kernel_size=5)
        self.conv_2 = torch.nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool1d(3, stride=2)
        self.linear_1 = torch.nn.Linear(in_features=380, out_features=128)
        self.linear_2 = torch.nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, u, v, t):
        u, v, t = u.unsqueeze(1), v.unsqueeze(1), t.unsqueeze(1)
        uv, ur, vr = u + v, u + t, v + t
        u_v, u_r, v_r = u * v, u * t, v * t
        x = torch.cat([u, v, t, uv, ur, vr, u_v, u_r, v_r], dim=1)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze()
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class VectorDataset(torch.utils.data.Dataset):
    def __init__(self, u=None, v=None, t=None, y=None, z=None):
        self.u = u
        self.v = v
        self.t = t
        self.y = y
        self.z = z

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def __getitem__(self, item):
        # node_feature = torch.cat([self.u[item], self.v[item]], dim=0)
        text_feature = self.tokenizer(self.t[item], truncation=True, padding="max_length", max_length=256)
        return torch.tensor(text_feature["input_ids"]), torch.tensor(text_feature["attention_mask"]), self.u[item], self.v[item],  \
               self.z[item], self.y[item]

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
    argument_parser.add_argument("-z_tr", "--vec_train", type=str, default="data/mini1k/train_vec.pt")
    argument_parser.add_argument("-z_te", "--vec_test", type=str, default="data/mini1k/test_vec.pt")
    argument_parser.add_argument("-d", "--device", type=str, default="cuda")
    argument_parser.add_argument("-e", "--epoch", type=int, default=50000)
    argument_parser.add_argument("-l", "--learning_rate", type=float, default=3e-5)
    argument_parser.add_argument("-s", "--save_dir", type=str, default="data/checkpoint")
    argument_parser.add_argument("-p", "--pretrained", type=str, default=None)
    args = argument_parser.parse_args()

    u_train, v_train = torch.load(args.u_train), torch.load(args.v_train)
    u_test, v_test = torch.load(args.v_test), torch.load(args.v_test)
    df = pd.read_csv(args.r_train)
    y_train = torch.from_numpy(np.array(df.Score.astype(int).tolist()) - 1)
    text_train = df.Text.tolist()
    vec_train, vec_test = torch.load(args.vec_train), torch.load(args.vec_test)
    df = pd.read_csv(args.r_test)
    y_test = torch.from_numpy(np.array(df.Score.astype(int).tolist()) - 1)
    text_test = df.Text.tolist()

    # net = JointClassifier(input_dim=896, num_classes=5)
    net = CNNClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lr=args.learning_rate, params=net.parameters())

    if args.pretrained is not None:
        net = net.load_state_dict(torch.load(args.pretrained, map_location=torch.device(args.device)))
        print(f"Found valid pretrained model and loaded from {args.pretrained}")

    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        VectorDataset(u=u_train, v=v_train, t=text_train, y=y_train, z=vec_train), shuffle=True, batch_size=16)
    test_loader = torch.utils.data.DataLoader(VectorDataset(u=u_test, v=v_test, t=text_test, y=y_test, z=vec_test),
                                              shuffle=True, batch_size=16)
    
    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    
    best_perf = 0.
    net = net.to(args.device)
    for epoch in range(args.epoch):
        # train
        net.train()
        loss_tr, cnt, acc_tr, f1_tr = 0., 0, 0., 0.
        for x, y, u, v, r, t in tqdm(train_loader, desc=f"Training {epoch + 1}/{args.epoch}"):
            x, y, u, v, r, t = x.to(args.device), y.to(args.device), u.to(args.device), v.to(args.device), r.to(args.device), t.to(args.device)
            p = net(u, v, r)
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
            for x, y, u, v, r, t in tqdm(test_loader, desc=f"Testing {epoch + 1}/{args.epoch}"):
                x, y, u, v, r, t = x.to(args.device), y.to(args.device), u.to(args.device), v.to(args.device), r.to(
                    args.device), t.to(args.device)
                p = net(u, v, r)
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

        writer.add_scalar('train_acc', acc_tr, epoch)
        writer.add_scalar('train_loss', loss_tr, epoch)
        writer.add_scalar('train_f1', f1_tr, epoch)
        writer.add_scalar('test_acc', acc_te, epoch)
        writer.add_scalar('test_loss', loss_te, epoch)
        writer.add_scalar('test_f1', f1_te, epoch)

    writer.close()
