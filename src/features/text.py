from tqdm import tqdm


def preprocess_df(df, tokenizer, stop_words):
    raw_docs = df['comment_text'].tolist()

    processed_docs = []
    for doc in tqdm(raw_docs):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs.append(" ".join(filtered))

    return processed_docs
