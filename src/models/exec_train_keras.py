import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, TensorBoard

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from src.utils import standard_parser, save_as_pickled_object, try_to_load_as_pickled_object_or_None
from src.data.load import load_embedding, load_data
from src.data.create import *
from src.features.text import preprocess_df
from src.features.embedding import prepare_embedding_matrix
from src.models import keras_util, keras_zoo
from modeldb.basic.ModelDbSyncerBase import *
import json


if __name__ == '__main__':
    options, args = standard_parser()

    with open(options.config, "rb") as f:
        configuration = json.load(f)

    np.random.seed(configuration["Seed"])
    max_nb_words = configuration["MaxNbWords"]

    model_name = create_model(configuration)

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
    X_val= sequence.pad_sequences(X_val, maxlen=max_seq_len)
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
    embedding_matrix, nb_words = prepare_embedding_matrix(word_index, max_nb_words, embeddings_index, size=embed_dim)

    # Note I'm wondering what would be the best way of specifying the model architecture... Commit every new
    # architecture?... Having a ModelDB tag for commit? Would require a large number of commits...

    model = keras_zoo.get_CNN_GRU_model(max_seq_len, nb_words, embedding_matrix, 300)

    write_path = configuration["ModelPath"] + model_name + "/"
    #define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2, verbose=1)
    auc_callback = keras_util.IntervalEvaluationROCAUCScore((X_val, y_val))
    tb = TensorBoard(log_dir=write_path)
    callbacks_list = [early_stopping, auc_callback]

    print(X_train.shape)
    print(X_val.shape)

    #model training
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list,
                     shuffle=False, verbose=1, validation_data=(X_val, y_val))


    with open(write_path + "model_architecture.json", 'w') as f:
        json.dump(model.to_json(), f)

    model.save_weights(write_path + "model_weight_final.h5")

    y_hat_test = model.predict(X_test)

    create_submission(test_df, y_hat_test, write_path, label_names)

    # add_modeldb_entry(config, saved_model_info)

