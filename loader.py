import time
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from torch.utils.data import Dataset
from torch_geometric.data import Data
from datetime import timedelta
from transformers import AutoTokenizer, AutoModel


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class AmazonFineFoodsReviews(object):
    def __init__(self, database_path: str):
        super(AmazonFineFoodsReviews, self).__init__()
        self.df = pd.read_csv(f"data/mini1k/test.csv")
        self.tf = pd.read_csv("data/mini1k/train.csv")
        print(f'Data:\n{self.df.describe()}')
        print(self.df.columns)

        self.udc, self.pdc = None, None

    @staticmethod
    def compress(x: List) -> (List, Dict):
        cnt, dc = 0, {}
        for i in range(len(x)):
            if x[i] not in dc:
                dc[x[i]] = cnt
                x[i] = cnt
                cnt = cnt + 1
            else:
                x[i] = dc[x[i]]

        return x, dc

    def build_graph(self):
        """
        Build graph from reviews
        :param text_feature:
        :return:
        """
        pivot = len(self.tf)
        user_ids, self.udc = AmazonFineFoodsReviews.compress(self.tf.UserId.tolist() + self.df.UserId.tolist())
        product_ids, self.pdc = AmazonFineFoodsReviews.compress(self.tf.ProductId.tolist() + self.df.ProductId.tolist())
        max_user_ids = max(user_ids)
        user_ids = np.array(user_ids)
        product_ids = np.array(product_ids) + max_user_ids + 1
        user_ids, product_ids = user_ids.reshape(1, user_ids.shape[0]), \
                                product_ids.reshape(1, product_ids.shape[0])

        edge_index = torch.tensor(np.concatenate((user_ids, product_ids), 0), dtype=torch.long)
        train_edge_index = edge_index[:, :pivot]
        # scores = self.df.Score.astype(int).tolist()
        train_y = torch.tensor(self.tf.Score.astype(int).tolist(), dtype=torch.long) - 1
        test_edge_index = edge_index[:, pivot:]
        test_y = torch.tensor(self.df.Score.astype(int).tolist(), dtype=torch.long) - 1
        # tuid, tpid = self.tf.UserId.tolist(), self.tf.ProductId.tolist()
        # tuid, tpid = np.array([self.udc[x] for x in tuid]), np.array([self.pdc[x] for x in tpid])
        # tuid, tpid = tuid.reshape(1, tuid.shape[0]), tpid.reshape(1, tpid.shape[0])
        # train_edge_index = torch.tensor(np.concatenate((tuid, tpid), 0), dtype=torch.long)

        # print(np.max(product_ids))
        # one_hot = torch.nn.functional.one_hot(torch.from_numpy(product_ids), num_classes=np.max(product_ids) + 1).squeeze(0)
        one_hot = torch.eye(np.max(product_ids) + 1)
        # print(one_hot.shape)
        return Data(x=one_hot, edge_index=edge_index, train_y=train_y, train_edge_index=train_edge_index, test_y=test_y,
                    test_edge_index=test_edge_index)


class TokenizedDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        super(TokenizedDataset, self).__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, it):
        return self.input_ids[it], self.attention_mask[it]

    def __len__(self):
        assert self.input_ids.shape == self.attention_mask.shape, ValueError("Dimension mismatch")
        return self.input_ids.shape[0]
