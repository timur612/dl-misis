from sklearn.feature_extraction.text import TfidfVectorizer

def make_tfidf_vectors(corpus):
    vectorizer = TfidfVectorizer(
        
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer