import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(df):
    vectorizer = TfidfVectorizer(lowercase=True, max_features=200, max_df=0.77, min_df=4, ngram_range= (1,3), step_words = "english")

    tfidf_matrix = vectorizer.fot_transform(df["clean_text"])

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    tfidf_df.head()