import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("data/Reviews.csv")
    scores = df.Score.astype(int).tolist()

    fig, ax = plt.subplots()

    # data = np.random.randint(1, 9, size=52)
    ax.hist(scores, bins=np.arange(0, 6) + 0.5, ec="k")
    ax.locator_params(axis='y', integer=True)

    plt.show()
    fig.savefig("data/stat.png", dpi=300)
