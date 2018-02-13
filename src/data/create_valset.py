import pandas as pd
import random
from math import floor


def train_val_split(df, perc):
    random.seed(966)

    pop = int(df.shape[0])

    k = floor(pop*perc)
    sample = random.sample(list(range(pop)), k)

    train = df[df.index.isin(sample)]
    val = df[~df.index.isin(sample)]

    return train, val


if __name__ == '__main__':
    data = pd.read_csv("/Users/kforest/Documents/workspace/toxiccomment/data/raw/train.csv")

    train, val = train_val_split(data, 0.8)

    train.to_csv("/Users/kforest/Documents/workspace/toxiccomment/data/processed/train_split_80.csv", index=False)
    val.to_csv("/Users/kforest/Documents/workspace/toxiccomment/data/processed/val_split_80.csv", index=False)