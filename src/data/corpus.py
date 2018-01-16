import pandas as pd


def create_complete_corpus(path_csv):
    """Create the complete corpus of both the train and test dataset.

    Specific to Toxic Comment dataset.
    """

    corpus = str()

    for i in path_csv:
        df = pd.read_csv(i)

        intra_corpus = df["comment_text"].str.cat(sep="\n")  # comment_text co is specific to this dataset column
        corpus += intra_corpus

    return corpus


if __name__ == '__main__':
    corpus = create_complete_corpus(["/Users/kforest/Documents/workspace/toxiccomment/data/raw/test.csv",
                                     "/Users/kforest/Documents/workspace/toxiccomment/data/raw/train.csv"])

    with open("/Users/kforest/Documents/workspace/toxiccomment/data/processed/corpus.txt", "w") as f:
        print(corpus, file=f)
