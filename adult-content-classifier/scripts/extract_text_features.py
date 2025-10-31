import pandas as pd
from textblob import TextBlob
import re
import os

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    return text.lower().strip()

def extract_text_features(caption):
    blob = TextBlob(caption)
    return {
        "char_count": len(caption),
        "word_count": len(caption.split()),
        "sentiment_polarity": blob.sentiment.polarity,
        "sentiment_subjectivity": blob.sentiment.subjectivity,
        "adult_keywords": sum([1 for w in caption.split() if w in ["hot","sexy","nude","kiss","bed","lust","erotic","nsfw"]])
    }

def main():
    df = pd.read_csv("data/captions.csv")
    features = []

    for i, row in df.iterrows():
        text = clean_text(row["caption"])
        feats = extract_text_features(text)
        feats["id"] = row["id"]
        features.append(feats)

    text_features = pd.DataFrame(features)
    os.makedirs("features", exist_ok=True)
    text_features.to_csv("features/text_features.csv", index=False)
    print("✅ Text features extracted and saved to features/text_features.csv")

if __name__ == "__main__":
    main()
