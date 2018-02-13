import codecs
from tqdm import tqdm
import numpy as np
import pandas as pd


def load_embedding(word_embedding, verbose=True):
    '''
    Load a word embedding based on a WordEmbedding config
    :param word_embedding: Dictionnary. Contain a Type. Type should be one of FastText, Glove, W2V. Depending on
        the type, can contain other values. For FastText, has to contain Path to binary and Encoding
    :param verbose: Boolean. Wether to to verbose of the loading
    :return: Dictionnary of words with their embedding
    '''
    POSSIBLE_TYPE = ["FastText", "Glove", "W2V", "Random"]
    if word_embedding['Type'] == "FastText":
        embeddings_index = {}
        f = codecs.open("/Users/kforest/workspace/toxiccomment/data/external/wiki.en.vec",
                        encoding=word_embedding["Encoding"])

        for line in tqdm(f, disable=not verbose):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        return embeddings_index

    elif word_embedding['Type'] in POSSIBLE_TYPE:
        raise NotImplementedError("")
    else:
        raise TypeError("Not a supported type of embedding")


def load_data(path, na_token='<NA>'):
    df = pd.read_csv(path, sep=',', header=0)
    df = df.fillna(na_token)

    return df
