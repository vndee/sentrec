import pickle
import numpy as np
import pandas as pd

if __name__ == '__main__':
    reviews = pd.read_csv('data/Reviews.csv')
    with open('data/edge_attr.pkl', 'rb') as stream:
        edge_attr = pickle.loads(stream.read())

    scores = reviews.Score.astype(int)

    classes = np.unique(scores)
    classes_count, min_c = [], scores.shape[0]
    for cls in classes:
        loc = np.where(scores == cls)[0]
        classes_count.append(loc)
        min_c = min(min_c, loc.shape[0])

    test_size = 0.1
    min_c = 1000
    new_train_indexes, new_test_indexes = None, None
    for it, cls in enumerate(classes_count):
        indexes = cls[: min_c]
        np.random.shuffle(indexes)

        new_train_indexes = np.atleast_1d(
            indexes[int(min_c * test_size):]) if new_train_indexes is None else np.concatenate(
            [new_train_indexes, indexes[int(min_c * test_size):]])
        new_test_indexes = np.atleast_1d(
            indexes[:int(min_c * test_size)]) if new_test_indexes is None else np.concatenate(
            [new_test_indexes, indexes[:int(min_c * test_size)]])

    train_reviews, test_reviews = reviews.iloc[new_train_indexes], reviews.iloc[new_test_indexes]
    edge_attr_train, edge_attr_test = edge_attr[new_train_indexes], edge_attr[new_test_indexes]

    train_reviews.to_csv("data/mini1k/train.csv")
    test_reviews.to_csv("data/mini1k/test.csv")
    with open("data/mini1k/train.vec", "wb") as stream:
        pickle.dump(edge_attr_train, stream)
    with open("data/mini1k/test.vec", "wb") as stream:
        pickle.dump(edge_attr_test, stream)
