import time
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from torch_geometric.data import Data
from datetime import timedelta
from transformers import AutoTokenizer, AutoModel


def chunks(lst, n, device):
    for i in range(0, lst["input_ids"].shape[0], n):
        yield {
            "input_ids": lst["input_ids"][i: i + n].to(device),
            "attention_mask": lst["attention_mask"][i: i + n].to(device)
        }


class AmazonFineFoodsReviews(object):
    def __init__(self, database_path: str):
        super(AmazonFineFoodsReviews, self).__init__()
        self.df = pd.read_csv(database_path)
        print(f'Data:\n{self.df.describe()}')
        print(self.df.columns)
        # self.df = self.df[:10000]

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

    def build_graph(self, text_feature=False, language_model_name='bert-base-cased', max_length=512, batch_size=16,
                    device='cpu'):
        """
        Build graph from reviews
        :param text_feature:
        :param language_model_name:
        :param max_length:
        :param batch_size:
        :param device:
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

        fig, ax = plt.subplots()

        ax.hist(scores, bins=np.arange(0, 6) + 0.5, ec="k")
        ax.locator_params(axis='y', integer=True)

        fig.savefig("data/build_graph_stat.png", dpi=300)

        if text_feature is True:
            return Data(x=torch.ones(1 + np.max(product_ids), 1), edge_index=edge_index,
                        y=torch.tensor(scores, dtype=torch.long) - 1), self.df.Text.tolist()

        return Data(x=torch.ones(1 + np.max(product_ids), 1), edge_index=edge_index,
                    y=torch.tensor(scores, dtype=torch.long) - 1)
