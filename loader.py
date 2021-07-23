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
        self.df = pd.read_csv(f"{database_path}.csv")
        self.tf = pd.read_csv("data/mini/train.csv")
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
        user_ids, self.udc = AmazonFineFoodsReviews.compress(self.df.UserId.tolist())
        product_ids, self.pdc = AmazonFineFoodsReviews.compress(self.df.ProductId.tolist())
        max_user_ids = max(user_ids)
        user_ids = np.array(user_ids)
        product_ids = np.array(product_ids) + max_user_ids + 1
        user_ids, product_ids = user_ids.reshape(1, user_ids.shape[0]), \
                                product_ids.reshape(1, product_ids.shape[0])

        edge_index = torch.tensor(np.concatenate((user_ids, product_ids), 0), dtype=torch.long)

        scores = self.df.Score.astype(int).tolist()

        train_y = torch.tensor(self.tf.Score.astype(int).tolist(), dtype=torch.long) - 1
        tuid, tpid = self.tf.UserId.tolist(), self.tf.ProductId.tolist()
        tuid, tpid = np.array([self.udc[x] for x in tuid]), np.array([self.pdc[x] for x in tpid])
        tpid = tpid + max_user_ids + 1
        tuid, tpid = tuid.reshape(1, tuid.shape[0]), tpid.reshape(1, tpid.shape[0])
        train_edge_index = torch.tensor(np.concatenate((tuid, tpid), 0), dtype=torch.long)

        return Data(x=torch.ones(1 + np.max(product_ids), 1), edge_index=edge_index,
                    y=torch.tensor(scores, dtype=torch.long) - 1, train_y=train_y, train_edge_index=train_edge_index)


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
