import pandas as pd

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from src.models.keras_zoo import get_CNN_model, get_ensemble_NN_model, get_CNN_LSTM_model, get_GRU_model
from src.features.config import NODEF_TOKEN, NULL_TOKEN, NODEF_TOKEN_VEC_VALUE, NULL_TOKEN_VEC_VALUE, SEQ_LENGTH, W2V_SIZE
from src.features.embedding import *
from gensim.models import KeyedVectors

from datetime import datetime
import os

model_dir = "/Users/kforest/Documents/workspace/toxiccomment/models/"
model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Name of the folder.
os.mkdir(model_dir + model_name)  # Create the folder of the model


train = pd.read_csv("../../data/processed/train_split_80.csv")
val = pd.read_csv("../../data/processed/val_split_80.csv")
test = pd.read_csv("../../data/raw/test.csv")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

list_sentences_train = train["comment_text"].fillna(NULL_TOKEN).values
y_train = train[list_classes].values

list_sentences_val = val["comment_text"].fillna(NULL_TOKEN).values
y_val = val[list_classes].values

list_sentences_test = test["comment_text"].fillna(NULL_TOKEN).values

en_model = KeyedVectors.load_word2vec_format('/Users/kforest/Documents/workspace/toxiccomment/data/external/wiki.en.vec')
indexes, space = get_embeding_space()

train["formatted_comment"] = train['comment_text'].apply(lambda x: embed_sentence(x, en_model))

X_t = np.asarray(train["formatted_comment"].tolist())

model = get_GRU_model(SEQ_LENGTH, space.shape[0], space.shape[1], space)
batch_size = 256
epochs = 1000

model.fit(X_t, y_train, batch_size=batch_size, epochs=epochs)

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

y_hat_val = model.predict(X_val)


sample_submission = pd.read_csv("../../data/raw/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv(model_dir + model_name + "/submission.csv", index=False)

sample_val = pd.read_csv("../../data/processed/val_split_80.csv")
sample_val[list_classes] = y_hat_val
sample_submission.to_csv(model_dir + model_name + "/val_y_hat.csv", index=False)
