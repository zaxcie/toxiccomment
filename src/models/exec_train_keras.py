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
from src.features.embedding import prepare_embedding_matrix
from src.models import keras_util, keras_zoo

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

    # TODO move everything to config file
    #training params
    batch_size = 256
    num_epochs = 8

    #model parameters
    num_filters = 64
    embed_dim = 300
    weight_decay = 1e-4

    #embedding matrix
    print('preparing embedding matrix...')
    embedding_matrix, nb_words = prepare_embedding_matrix(word_index, max_nb_words, embeddings_index, size=embed_dim)

    #CNN architecture
    # TODO move to Keras Zoo
    # Note I'm wondering what would be the best way of specifying the model architecture... Commit every new
    # architecture?... Having a ModelDB tag for commit? Would require a large number of commits...

    model = keras_zoo.get_CNN_model(nb_words, 300, embedding_matrix, 168)


    #define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, verbose=1)
    auc_callback = keras_util.IntervalEvaluationROCAUCScore((X_val, y_val))
    callbacks_list = [early_stopping, auc_callback]

    #model training
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                     shuffle=False, verbose=1, validation_data=(X_val, y_val))

    y_test = model.predict(X_test)


    #create a submission
    submission_df = pd.DataFrame(columns=['id'] + label_names)
    submission_df['id'] = test_df['id'].values
    submission_df[label_names] = y_test
    submission_df.to_csv("../cnn_fasttext_submission.csv", index=False)

