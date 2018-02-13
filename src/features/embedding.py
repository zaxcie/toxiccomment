import numpy as np


def prepare_embedding_matrix(word_index, max_nb_words, embeddings_index, size=300):
    nb_words = min(max_nb_words, len(word_index))
    embedding_matrix = np.zeros((nb_words, size))

    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, nb_words
