import pandas as pd

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from src.models.keras_zoo import get_CNN_model, get_ensemble_NN_model, get_CNN_LSTM_model

from datetime import datetime
import os

model_dir = "/Users/kforest/Documents/workspace/toxiccomment/models/"
model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Name of the folder.
os.mkdir(model_dir + model_name)  # Create the folder of the model

max_features = 50000
maxlen = 200
number_filters = 200

train = pd.read_csv("data/processed/train_split_80.csv")
val = pd.read_csv("data/processed/val_split_80.csv")
test = pd.read_csv("data/raw/test.csv")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

list_sentences_train = train["comment_text"].fillna("zaxcie").values
y_train = train[list_classes].values

list_sentences_val = val["comment_text"].fillna("zaxcie").values
y_val = val[list_classes].values

list_sentences_test = test["comment_text"].fillna("zaxcie").values


tokenizer_train = text.Tokenizer(num_words=max_features)
tokenizer_train.fit_on_texts(list(list_sentences_train))

tokenizer_val = text.Tokenizer(num_words=max_features)
tokenizer_val.fit_on_texts(list(list_sentences_val))

list_tokenized_train = tokenizer_train.texts_to_sequences(list_sentences_train)
list_tokenized_val = tokenizer_train.texts_to_sequences(list_sentences_val)
list_tokenized_test = tokenizer_train.texts_to_sequences(list_sentences_test)

X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_val = sequence.pad_sequences(list_tokenized_val, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


# X_t = X_t.reshape((X_t.shape[0], 1, X_t.shape[1]))
# X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
# X_te = X_te.reshape((X_te.shape[0], 1, X_te.shape[1]))

model = get_CNN_LSTM_model(maxlen, max_features, number_filters)
batch_size = 64
epochs = 1000


file_path = model_dir + model_name + "/weights_base.best.hdf5"

tbCall = TensorBoard(log_dir=model_dir + model_name + "/", histogram_freq=0,
                     write_graph=False, write_images=True)
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=2)


callbacks_list = [checkpoint, early, tbCall]
model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
          callbacks=callbacks_list)

model.load_weights(file_path)

y_test = model.predict(X_te)


sample_submission = pd.read_csv("data/raw/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv(model_dir + model_name + "/submission.csv", index=False)
