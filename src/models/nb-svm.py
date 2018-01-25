import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datetime import datetime
import os
from src.utils import write_comment, get_standard_parser
import re
import string



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
subm = pd.read_csv('/Users/kforest/Documents/workspace/toxiccomment/data/raw/sample_submission.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
label_cols_ = ['toxic_', 'severe_toxic_', 'obscene_', 'threat_', 'insult_', 'identity_hate_']
train['none'] = 1-train[label_cols].max(axis=1)
val['none'] = 1-val[label_cols].max(axis=1)
train.describe()

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
val[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s): return re_tok.sub(r' \1 ', s).split()


n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])
val_term_doc = vec.transform(val[COMMENT])

x = trn_term_doc
test_x = test_term_doc
val_x = val_term_doc


def pr(y_i, y):
    p = x[y == y_i].sum(0)
    return (p+1) / ((y == y_i).sum()+1)


def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


preds = np.zeros((len(test), len(label_cols)))
val_preds = np.zeros((len(val), len(label_cols)))
val_probs = np.zeros((len(val), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m, r = get_mdl(train[j])
    preds[:, i] = m.predict_proba(test_x.multiply(r))[:, 1]
    val_probs[:, i] = m.predict_proba(val_x.multiply(r))[:, 1]
    val_preds[:, i] = m.predict(val_x.multiply(r))


submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv(MODEL_DIR + "/" + model_name + '/submission.csv', index=False)

valsid = pd.DataFrame({'id': val["id"]})
val_hy_hat = pd.concat([valsid, pd.DataFrame(val_probs, columns=label_cols)], axis=1)
val_hy_hat = pd.concat([val_hy_hat, pd.DataFrame(val_preds, columns=label_cols_)], axis=1)
val_hy_hat.to_csv(MODEL_DIR + "/" + model_name + '/val_out.csv', index=False)
