import codecs
from tqdm import tqdm
import numpy as np


def load_embedding(word_embedding):
    '''
    Load a word embedding based on a WordEmbedding config.
    :param word_embedding: Dictionnary. Contain a Type, a Path and an Encoding. Type should be one of FastText, Glove, W2V
    :return: Dictionnary of words with their embedding
    '''
    POSSIBLE_TYPE = ["FastText", "Glove", "W2V"]
    if word_embedding['Type'] == "FastText":
        embeddings_index = {}
        f = codecs.open("/Users/kforest/workspace/toxiccomment/data/external/wiki.en.vec", encoding=word_embedding[])
        for line in tqdm(f):
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