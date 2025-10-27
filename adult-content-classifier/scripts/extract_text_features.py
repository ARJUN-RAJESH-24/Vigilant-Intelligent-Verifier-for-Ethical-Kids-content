import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def extract_tfidf(caption_file, out_csv):
    df = pd.read_csv(caption_file)
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df['caption'].astype(str))
    X = normalize(X)
    out = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    out.insert(0, 'id', df['id'])
    out.to_csv(out_csv, index=False)

if __name__ == "__main__":
    import os
    os.makedirs('features', exist_ok=True)
    extract_tfidf('data/captions.csv','features/text_features.csv')
