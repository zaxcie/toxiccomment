import codecs
from tqdm import tqdm
import numpy as np
import pandas as pd
from src.utils import save_as_pickled_object, try_to_load_as_pickled_object_or_None


def load_embedding(word_embedding, verbose=True):
    '''
    Load a word embedding based on a WordEmbedding config
    :param word_embedding: Dictionnary. Contain a Type. Type should be one of FastText, Glove, W2V. Depending on
        the type, can contain other values. For FastText, has to contain Path to binary and Encoding
    :param verbose: Boolean. Wether to to verbose of the loading
    :return: Dictionnary of words with their embedding
    '''

    # TODO add more documentation regarding embedding possibility

    POSSIBLE_TYPE = ["Processed", "FastText", "Glove", "W2V", "Random"]

    if word_embedding['Type'] == "Processed":
        embeddings_index = try_to_load_as_pickled_object_or_None(word_embedding["Path"])

        return embeddings_index

    if word_embedding['Type'] == "FastText":
        embeddings_index = {}
        f = codecs.open(word_embedding["Path"],
                        encoding=word_embedding["Encoding"])

        for line in tqdm(f, disable=not verbose):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        if word_embedding['Save']:
            save_as_pickled_object(embeddings_index, word_embedding['OutputFile'])

        return embeddings_index

    elif word_embedding['Type'] in POSSIBLE_TYPE:
        raise NotImplementedError("")
    else:
        raise TypeError("Not a supported type of embedding")


def load_data(path, na_token='<NA>'):
    df = pd.read_csv(path, sep=',', header=0)
    df = df.fillna(na_token)

    return df
