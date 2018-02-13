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
import json

if __name__ == '__main__':

    options, args = standard_parser()

    with open(options.config, "rb") as f:
        configuration = json.load(f)

    np.random.seed(configuration["Seed"])
    max_nb_words = configuration["MaxNbWords"]

    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])


    #load embeddings
    embeddings_index = load_embedding(configuration["WordEmbedding"])

    #load data
    train_df = load_data(configuration["TrainDataPath"])
    val_df = load_data(configuration["ValDataPath"])
    test_df = load_data(configuration["TestDataPath"])

    label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y_train = train_df[label_names].values

    train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))
    max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)

    raw_docs_train = train_df['comment_text'].tolist()
    raw_docs_val = val_df['comment_text'].tolist()
    raw_docs_test = test_df['comment_text'].tolist()
    num_classes = len(label_names)

    print("pre-processing train data...")
    processed_docs_train = []
    for doc in tqdm(raw_docs_train):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs_train.append(" ".join(filtered))
    #end for

    processed_docs_test = []
    for doc in tqdm(raw_docs_test):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs_test.append(" ".join(filtered))
    #end for

    print("tokenizing input data...")
    tokenizer = Tokenizer(num_words=max_nb_words, lower=True, char_level=False)
    tokenizer.fit_on_texts(processed_docs_train + processed_docs_test)  # leaky
    word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
    word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
    word_index = tokenizer.word_index
    print("dictionary size: ", len(word_index))

    #pad sequences
    word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
    word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

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
    hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                     validation_split=0.1, shuffle=True, verbose=1)

    y_test = model.predict(word_seq_test)

    #create a submission
    submission_df = pd.DataFrame(columns=['id'] + label_names)
    submission_df['id'] = test_df['id'].values
    submission_df[label_names] = y_test
    submission_df.to_csv("../cnn_fasttext_submission.csv", index=False)

