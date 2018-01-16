from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Conv2D, Embedding, Dropout, Activation
from keras.layers import Bidirectional, MaxPooling1D, MaxPooling2D, Reshape, Flatten, concatenate, BatchNormalization, LSTM, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, backend


def get_CNN_model(maxlen, max_features, embed_size, number_filters):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x1 = Conv2D(number_filters, (3, embed_size))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((int(int(x1.shape[2]) / 1.5), 1))(x1)
    x1 = Flatten()(x1)

    x2 = Conv2D(number_filters, (4, embed_size))(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('elu')(x2)
    x2 = MaxPooling2D((int(int(x2.shape[2]) / 1.5), 1))(x2)
    x2 = Flatten()(x2)

    x3 = Conv2D(number_filters, (5, embed_size))(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D((int(int(x3.shape[2]) / 1.5), 1))(x3)
    x3 = Flatten()(x3)

    x4 = Conv2D(number_filters, (6, embed_size))(x)
    x4 = BatchNormalization()(x4)
    x4 = Activation('elu')(x4)
    x4 = MaxPooling2D((int(int(x4.shape[2]) / 1.5), 1))(x4)
    x4 = Flatten()(x4)

    x5 = Conv2D(number_filters, (7, embed_size))(x)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    x5 = MaxPooling2D((int(int(x5.shape[2]) / 1.5), 1))(x5)
    x5 = Flatten()(x5)

    x = concatenate([x1, x2, x3, x4, x5])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def get_LSTM_model(maxlen, max_features, embed_size=64):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def get_CNN_LSTM_model(maxlen, max_features, number_filters, embed_size=64):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)

    x = Conv2D(number_filters, (5, embed_size))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((int(int(x.shape[2]) / 1.5), 1))(x)
    x = Flatten()(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def get_ensemble_NN_model(maxlen, max_features, number_filters, embed_size=64):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)

    x1 = Conv2D(number_filters, (5, embed_size))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((int(int(x1.shape[2]) / 1.5), 1))(x1)
    x1 = Flatten()(x1)

    x1 = Bidirectional(LSTM(128, return_sequences=True))(x1)
    x1 = GlobalMaxPool1D()(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(50, activation="relu")(x)

    x2 = Bidirectional(LSTM(128, return_sequences=True))(x)
    x2 = GlobalMaxPool1D()(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Dense(50, activation="relu")(x2)

    x3 = Conv2D(number_filters, (5, embed_size))(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D((int(int(x3.shape[2]) / 1.5), 1))(x3)
    x3 = Flatten()(x3)

    x4 = Conv2D(number_filters, (6, embed_size))(x)
    x4 = BatchNormalization()(x4)
    x4 = Activation('elu')(x4)
    x4 = MaxPooling2D((int(int(x4.shape[2]) / 1.5), 1))(x4)
    x4 = Flatten()(x4)

    x5 = Conv2D(number_filters, (7, embed_size))(x)
    x5 = BatchNormalization()(x5)
    x5 = Activation('relu')(x5)
    x5 = MaxPooling2D((int(int(x5.shape[2]) / 1.5), 1))(x5)
    x5 = Flatten()(x5)

    x = concatenate([x1, x2, x3, x4, x5])
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model