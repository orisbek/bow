from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def fit_transform_train_test(train_texts, test_texts):
    bow = CountVectorizer(ngram_range=(1, 1), min_df=1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
    X_bow_train = bow.fit_transform(train_texts)
    X_bow_test = bow.transform(test_texts)
    X_tfidf_train = tfidf.fit_transform(train_texts)
    X_tfidf_test = tfidf.transform(test_texts)
    return {
        'bow': (bow, X_bow_train, X_bow_test),
        'tfidf': (tfidf, X_tfidf_train, X_tfidf_test),
    }
