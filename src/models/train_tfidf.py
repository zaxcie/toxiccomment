import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.utils import standard_parser
from src.data.load import load_data
from src.data.create import *
from src.features.tfidf import tfidf_features
from sklearn.feature_extraction.text import TfidfVectorizer


label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

options, args = standard_parser()

with open(options.config, "rb") as f:
    configuration = json.load(f)

np.random.seed(configuration["Seed"])
max_nb_words = configuration["MaxNbWords"]

model_name = create_model(configuration)

train_df = load_data(configuration["TrainDataPath"])
val_df = load_data(configuration["ValDataPath"])
test_df = load_data(configuration["TestDataPath"])

corpus = pd.concat([train_df["comment_text"], val_df["comment_text"], test_df["comment_text"]])

print("fit tfidf")
char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer="char",
        ngram_range=(1,5),
        max_features=25000)

char_vectorizer.fit(corpus)

print(" vectorize")
train_features = char_vectorizer.transform(train_df["comment_text"])
val_features = char_vectorizer.transform(val_df["comment_text"])
test_features = char_vectorizer.transform(test_df["comment_text"])

#train_features = hstack([train_char_features, train_word_features])
#test_features = hstack([test_char_features, test_word_features])

losses = []
predictions = {'id': test_df['id']}
for class_name in label_names:
    print("Start training")
    train_target = train_df[class_name]
    classifier = LogisticRegression(solver='sag', verbose=1, n_jobs=-1)

    cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=3, scoring='roc_auc'))
    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))

    classifier.fit(train_features, train_target)
    predictions[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(losses)))

# submission = pd.DataFrame.from_dict(predictions)
# submission.to_csv('submission.csv', index=False)
