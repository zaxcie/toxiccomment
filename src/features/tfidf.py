from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_features(corpus, df, max_feature=25000, analyzer="char", ngram=(1, 5)):
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer=analyzer,
        ngram_range=ngram,
        max_features=max_feature)
    char_vectorizer.fit(corpus)
    features = char_vectorizer.transform(df["comment_text"])

    return features
