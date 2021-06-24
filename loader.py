import time
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from torch.utils.data import Dataset
from torch_geometric.data import Data
from datetime import timedelta
from transformers import AutoTokenizer, AutoModel


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class AmazonFineFoodsReviews(object):
    def __init__(self, database_path: str, test_path=None):
        super(AmazonFineFoodsReviews, self).__init__()
        self.df = pd.read_csv(f"{database_path}.csv")
        self.tdf = pd.read_csv(f"{test_path}.csv")
        self.pivot = len(self.df)
        self.df["test"] = 0
        self.tdf["test"] = 1
        self.df = pd.concat([self.df, self.tdf])
        print(f'Data:\n{self.df.describe()}')
        print(self.df.columns)

    @staticmethod
    def compress(x: List) -> List:
        cnt, dc = 0, {}
        for i in range(len(x)):
            if x[i] not in dc:
                dc[x[i]] = cnt
                x[i] = cnt
                cnt = cnt + 1
            else:
                x[i] = dc[x[i]]

        return x

    def build_graph(self, text_feature=False):
        """
        Build graph from reviews
        :param text_feature:
        :return:
        """
        user_ids = AmazonFineFoodsReviews.compress(self.df.UserId.tolist())
        product_ids = AmazonFineFoodsReviews.compress(self.df.ProductId.tolist())
        max_user_ids = max(user_ids)
        user_ids = np.array(user_ids)
        product_ids = np.array(product_ids) + max_user_ids + 1
        user_ids, product_ids = user_ids.reshape(1, user_ids.shape[0]), \
                                product_ids.reshape(1, product_ids.shape[0])

        edge_index = torch.tensor(np.concatenate((user_ids, product_ids), 0), dtype=torch.long)

        # self.df.Score = self.df.Score.apply(lambda x: 0 if x < 3 else 1 if x == 3 else 2)
        scores = self.df.Score.astype(int).tolist()

        if text_feature is True:
            return Data(x=torch.ones(1 + np.max(product_ids), 1), edge_index=edge_index,
                        y=torch.tensor(scores, dtype=torch.long) - 1), self.df.Text.tolist(), self.pivot

        return Data(x=torch.ones(1 + np.max(product_ids), 1), edge_index=edge_index,
                    y=torch.tensor(scores, dtype=torch.long) - 1), self.pivot


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
