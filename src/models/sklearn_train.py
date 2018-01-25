import pandas as pd
import numpy as np
from sklearn import *
from datetime import datetime
import os
from src.utils import get_standard_parser, write_comment

model_dir = "../../models/"
model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Name of the folder.
os.mkdir(model_dir + model_name)  # Create the folder of the model

# TODO don't hard code model dir and data dir. Ok for now.

options, args = get_standard_parser()
DATA_DIR = options.data_dir
MODEL_DIR = options.model_dir
MODEL_COMMENT = options.comment

if MODEL_COMMENT == "":
    MODEL_COMMENT = input("Comment about the model to produce ")

write_comment(MODEL_COMMENT, MODEL_DIR, model_name)

print("Load data")

train = pd.read_csv("/Users/kforest/Documents/workspace/toxiccomment/data/processed/train_split_80.csv")
test = pd.read_csv("/Users/kforest/Documents/workspace/toxiccomment/data/raw/test.csv")
val = pd.read_csv("/Users/kforest/Documents/workspace/toxiccomment/data/processed/val_split_80.csv")

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

coly = [c for c in train.columns if c not in ['id', 'comment_text']]
y_train = train[coly]
y_val = val[coly]
tid = test['id'].values

df = pd.concat([train['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow_train = train.shape[0]
nrow_val = val.shape[0]

nrow_reach_val = nrow_train + nrow_val

tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', max_features=50000)
data = tfidf.fit_transform(df)

train_df = data[:nrow_train]
val_df = data[nrow_train:nrow_reach_val]
test_df = data[nrow_reach_val:]

model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3, verbose=1,
                                      n_estimators=80)
model.fit(train_df, y_train)

val_output = model.predict(val_df)
test_output = model.predict_proba(test_df)

for i in range(6):
    loss = np.asarray(metrics.log_loss(y_val[list_classes[i]], val_output[i])).mean()

print(loss)
