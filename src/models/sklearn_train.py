import pandas as pd
import numpy as np
from sklearn import *
from datetime import datetime
import os


model_dir = "/Users/kforest/Documents/workspace/toxiccomment/models/"
model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Name of the folder.
os.mkdir(model_dir + model_name)  # Create the folder of the model

print("Load data")

train = pd.read_csv("/Users/kforest/Documents/workspace/toxiccomment/data/processed/train_split_80.csv")
val = pd.read_csv("/Users/kforest/Documents/workspace/toxiccomment/data/processed/val_split_80.csv")
test = pd.read_csv("/Users/kforest/Documents/workspace/toxiccomment/data/raw/test.csv")

max_features = 20000

y_col = [c for c in train.columns if c not in ['id', 'comment_text']]
y_train = train[y_col]
y_val = train[y_col]

tid = test['id'].values

df = pd.concat([train['comment_text'], val['comment_text'], test['comment_text']], axis=0)
df = df.fillna("unknown")
nrow = train.shape[0]
val_nrow = val.shape[0]

print("TFIDF")

tfidf = feature_extraction.text.TfidfVectorizer(stop_words='english', max_features=max_features)
data = tfidf.fit_transform(df)

print("Training")

model = ensemble.ExtraTreesClassifier(n_jobs=1, random_state=966, verbose=1)
model.fit(data[:nrow], y_train)

print(1 - model.score(data[:nrow], y_train))

# Validation
print("Validation")
val_out = pd.DataFrame(model.predict(data[nrow:(nrow + val_nrow)]))
val_out.columns = y_col

mean_cross_entropy = []
for c in y_col:
    val_out[c] = val_out[c].clip(0 + 1e12, 1 - 1e12)
    mean_cross_entropy.append(metrics.log_loss(y_val[c], val_out[c]))

mean_cross_entropy = np.asarray(mean_cross_entropy)

print(mean_cross_entropy.mean())

# Submission
print("Submission generation")
sub = pd.DataFrame(model.predict(data[(nrow + val_nrow - 1):]))
sub.columns = y_col
sub['id'] = tid

for c in y_col:
    sub[c] = sub[c].clip(0+1e12, 1-1e12)

sub.to_csv(model_dir + model_name + '/submission.csv', index=False)
