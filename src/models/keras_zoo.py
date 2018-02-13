from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Conv2D, Embedding, Dropout, Activation, TimeDistributed, Conv1D, Concatenate
from keras.layers import Bidirectional, MaxPooling1D, MaxPooling2D, Reshape, Flatten, concatenate
from keras.layers import BatchNormalization, LSTM, GlobalMaxPool1D, GRU, GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers, backend


def get_CNN_model(nb_words, embed_size, embedding_matrix, seq_len, num_classes=6):
    # hyper-parameters
    # Note I'm trying to figuring out a better way to save model architecture and connect with ModelDB.
    # For now, the best way seems to hard code them here and attach partial model architecture to ModelDB and
    # save complete model architecture to model folder.
    num_filters = 64
    weight_decay = 1e-4

    model = Sequential()
    model.add(Embedding(nb_words, embed_size,
                        weights=[embedding_matrix], input_length=seq_len, trainable=False))
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

    return model


def get_LSTM_model(maxlen, max_features, embed_size=64):
    inp = Input(shape=(maxlen, ))
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
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


def get_GRU_model(maxlen, max_features, embed_size, embed_weights, train_embed=False):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, trainable=train_embed)(inp)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(GRU(64, return_sequences=False))(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    return model


def get_CNN_LSTM_model(maxlen, max_features, number_filters, embed_size=300):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Dropout(0.2)(x)
    convs = []

    for i, k_size in enumerate([1, 3, 5, 7]):
        conv = Conv1D(filters=128, kernel_size=k_size, padding='same',
                      activation='relu')(x)
        convs.append(conv)

    x = Concatenate()(convs)
    x = MaxPooling1D()(x)

    x = Conv1D(filters=64, kernel_size=3, padding='same',
               activation='relu')(x)
    x = MaxPooling1D()(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def get_ensemble_NN_model(maxlen, max_features, number_filters, embed_size=64):
    inp = Input(shape=(maxlen, 1, ))
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

    x3 = Conv2D(number_filters, (2, embed_size))(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = MaxPooling2D((int(int(x3.shape[2]) / 1.5), 1))(x3)
    x3 = Flatten()(x3)

    x4 = Conv2D(number_filters, (4, embed_size))(x)
    x4 = BatchNormalization()(x4)
    x4 = Activation('elu')(x4)
    x4 = MaxPooling2D((int(int(x4.shape[2]) / 1.5), 1))(x4)
    x4 = Flatten()(x4)

    x5 = Conv2D(number_filters, (6, embed_size))(x)
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


def get_CNN_GRU_model(maxlen, max_features, number_filters, embed_size=300):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Dropout(0.2)(x)
    convs = []

    for i, k_size in enumerate([1, 3, 5, 7]):
        conv = Conv1D(filters=128, kernel_size=k_size, padding='same',
                      activation='relu')(x)
        convs.append(conv)

    x = Concatenate()(convs)
    x = MaxPooling1D()(x)

    x = Conv1D(filters=64, kernel_size=3, padding='same',
               activation='relu')(x)
    x = MaxPooling1D()(x)

    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


