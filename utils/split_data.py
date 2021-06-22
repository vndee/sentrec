import os
import pickle
import numpy as np
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv("data/Reviews.csv")

    index = np.arange(len(df))
    np.random.shuffle(index)

    pivot = 500000
    train_df = df.iloc[index[:pivot]]
    test_df = df.iloc[index[pivot:]]

    print(f"Length train df: {len(train_df)}")
    print(f"Length test df: {len(test_df)}")

    train_df.to_csv("data/train.csv")
    test_df.to_csv("data/test.csv")

    del index, train_df, test_df
    with open("data/edge_attr.pkl", "rb") as stream:
        arr = pickle.loads(stream.read())

    with open("data/train.vec", "wb") as stream:
        pickle.dump(arr[:pivot], stream)

    with open("data/test.vec", "wb") as stream:
        pickle.dump(arr[pivot:], stream)

