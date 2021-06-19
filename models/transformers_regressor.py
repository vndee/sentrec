import random
import torch
import pickle
import numpy as np
import torch.nn as nn
from transformers import AutoModel
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class TransformerRegressor(nn.Module):
    def __init__(self, language_model_shortcut='bert-base-cased', hidden_dim=768, inner_dim=128):
        super(TransformerRegressor, self).__init__()
        # self.backbone = AutoModel.from_pretrained(language_model_shortcut)
        self.linear1 = nn.Linear(hidden_dim, inner_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(inner_dim, 1)

    def forward(self, x):
        # x = self.backbone({
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "token_type_ids": token_type_ids
        # })

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0], ValueError("Dataset dimensions wrong")
        self.__x = x
        self.__y = y

    def __getitem__(self, it):
        return self.__x[it], self.__y[it]

    def __len__(self):
        return self.__x.shape[0]


# test
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    with open("data/graph.pkl", "rb") as stream:
        graph = pickle.loads(stream.read())

    with open("data/edge_attr.pkl", "rb") as stream:
        edge_attr = pickle.loads(stream.read())

    target = graph.y - 1
    target = ((target - torch.min(target)) / (torch.max(target) - torch.min(target))).unsqueeze(-1)

    lr = 0.001
    net = TransformerRegressor()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    test_size = 0.2
    batch_size, num_workers = 4, 4
    data_index = np.arange(target.shape[0])
    np.random.shuffle(data_index)
    val_index = data_index[: int(test_size * data_index.shape[0])]
    train_index = data_index[int(test_size * data_index.shape[0]):]

    dataset = MyDataset(edge_attr, target)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_index), pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_index), pin_memory=True,
                            num_workers=num_workers)

    max_epoch = 5
    losses = [[] for x in range(5)]
    for epoch in range(max_epoch):
        train_loss, val_loss = 0., 0.

        # training
        net.train()
        for x, y in train_loader:
            z = net(x)
            loss = criterion(z, y)
            loss.backward()
            train_loss += loss.item()
            losses[epoch].append(loss.item())

        # validating
        net.eval()
        with torch.no_grad():
            for x, y in val_loader:
                z = net(x)
                loss = criterion(z, y)
                val_loss += loss.item()

        train_loss, val_loss = train_loss / len(train_loader), val_loss / len(val_loader)
        print(f"Epoch {epoch + 1:04d}/{max_epoch:04d}: train: {train_loss:.8f} - val: {val_loss:.8f}")

    for i in losses:
        print(i)