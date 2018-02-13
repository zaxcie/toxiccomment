import numpy as np
import pandas as pd
import seaborn as sns

from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import codecs
from src.utils import standard_parser
from src.data.load import load_embedding, load_data
from src.features.text import preprocess_df

import json

if __name__ == '__main__':

    options, args = standard_parser()

    with open(options.config, "rb") as f:
        configuration = json.load(f)

    np.random.seed(configuration["Seed"])
    max_nb_words = configuration["MaxNbWords"]

    # TODO define somewhere else. Not sure where...
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

    # load embeddings
    embeddings_index = load_embedding(configuration["WordEmbedding"])

    # load data
    train_df = load_data(configuration["TrainDataPath"])
    val_df = load_data(configuration["ValDataPath"])
    test_df = load_data(configuration["TestDataPath"])

    label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    num_classes = len(label_names)
    y_train = train_df[label_names].values
    y_val = val_df[label_names].values

    train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))
    max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)

    processed_comments_train = preprocess_df(train_df, tokenizer=tokenizer, stop_words=stop_words)
    processed_comments_val = preprocess_df(val_df, tokenizer=tokenizer, stop_words=stop_words)
    processed_comments_test = preprocess_df(test_df, tokenizer=tokenizer, stop_words=stop_words)

    tokenizer = Tokenizer(num_words=max_nb_words, lower=True, char_level=False)
    tokenizer.fit_on_texts(processed_comments_train + processed_comments_val + processed_comments_test)
    X_train = tokenizer.texts_to_sequences(processed_comments_train)
    X_val = tokenizer.texts_to_sequences(processed_comments_val)
    X_test = tokenizer.texts_to_sequences(processed_comments_test)
    word_index = tokenizer.word_index

    #pad sequences
    X_train = sequence.pad_sequences(X_train, maxlen=max_seq_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_seq_len)

    #training params
    batch_size = 256
    num_epochs = 8

    #model parameters
    num_filters = 64
    embed_dim = 300
    weight_decay = 1e-4

    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    nb_words = min(max_nb_words, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_dim))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)

    print(embedding_matrix)
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    #CNN architecture
    print("training CNN ...")
    model = Sequential()
    model.add(Embedding(nb_words, embed_dim,
              weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(num_classes, activation='sigmoid'))  # multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    #define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
    callbacks_list = [early_stopping]

    #model training
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                     validation_split=0.1, shuffle=True, verbose=1)

    y_test = model.predict(X_test)

    #create a submission
    submission_df = pd.DataFrame(columns=['id'] + label_names)
    submission_df['id'] = test_df['id'].values
    submission_df[label_names] = y_test
    submission_df.to_csv("../cnn_fasttext_submission.csv", index=False)

