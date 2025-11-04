import pandas as pd
from textblob import TextBlob
import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

CAPTIONS_PATH = "data/captions.csv"
OUTPUT_PATH = "features/text_features.csv"
TFIDF_VECTOR_PATH = "features/tfidf_features.csv"
TFIDF_MODEL_PATH = "models/tfidf_vectorizer.pkl"

ADULT_KEYWORDS = [
    "hot", "sexy", "nude", "naked", "kiss", "bed", "lust", "erotic", "nsfw",
    "sex", "porn", "xxx", "adult", "intimate", "sensual", "seductive", 
    "provocative", "explicit", "mature", "bikini", "lingerie"
]

VIOLENCE_KEYWORDS = [
    "kill", "blood", "murder", "death", "violent", "fight", "attack", 
    "weapon", "gun", "knife", "war", "combat"
]

def clean_text(text):
    """Clean text by removing URLs, mentions, hashtags, and special characters."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\S+|#\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s.,!?]+", "", text)
    text = " ".join(text.split())
    
    return text.lower().strip()

def extract_text_features(caption):
    """Extract structural and sentiment text features from caption."""
    if not caption or len(caption) == 0:
        return {
            "char_count": 0, "word_count": 0, "avg_word_length": 0.0,
            "sentiment_polarity": 0.0, "sentiment_subjectivity": 0.0,
            "adult_keywords_count": 0, "violence_keywords_count": 0,
            "uppercase_ratio": 0.0, "punctuation_count": 0, "exclamation_count": 0,
            "question_count": 0, "number_count": 0,
        }
    
    char_count = len(caption)
    words = caption.split()
    word_count = len(words)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0
    
    try:
        blob = TextBlob(caption)
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity
    except:
        sentiment_polarity = 0.0
        sentiment_subjectivity = 0.0
    
    caption_lower = caption.lower()
    adult_count = sum(1 for keyword in ADULT_KEYWORDS if keyword in caption_lower)
    violence_count = sum(1 for keyword in VIOLENCE_KEYWORDS if keyword in caption_lower)
    
    uppercase_count = sum(1 for c in caption if c.isupper())
    uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0.0
    
    punctuation_count = sum(1 for c in caption if c in ".,;:!?")
    exclamation_count = caption.count("!")
    question_count = caption.count("?")
    number_count = sum(1 for c in caption if c.isdigit())
    
    return {
        "char_count": char_count, "word_count": word_count, "avg_word_length": float(avg_word_length),
        "sentiment_polarity": float(sentiment_polarity), "sentiment_subjectivity": float(sentiment_subjectivity),
        "adult_keywords_count": adult_count, "violence_keywords_count": violence_count,
        "uppercase_ratio": float(uppercase_ratio), "punctuation_count": punctuation_count,
        "exclamation_count": exclamation_count, "question_count": question_count,
        "number_count": number_count,
    }

def generate_tfidf_features(df):
    """Generates TF-IDF features from cleaned captions and saves them."""
    if len(df) == 0:
        print("⚠️ Warning: Empty DataFrame, skipping TF-IDF generation.")
        return
        
    print("✨ Generating TF-IDF features...")
    
    tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_caption']).toarray()
    
    feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=feature_names)
    
    tfidf_df['id'] = df['id']
    cols = ['id'] + [col for col in tfidf_df.columns if col != 'id']
    tfidf_df = tfidf_df[cols]
    
    tfidf_df.to_csv(TFIDF_VECTOR_PATH, index=False)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(tfidf_vectorizer, TFIDF_MODEL_PATH)
    print(f"✅ TF-IDF vectors saved to {TFIDF_VECTOR_PATH}")
    print(f"✅ TF-IDF model saved to {TFIDF_MODEL_PATH}")
    print(f"📈 Extracted {len(tfidf_df)} samples with {tfidf_matrix.shape[1]} TF-IDF features")


def main():
    """Main function to extract text features from captions."""
    if not os.path.exists(CAPTIONS_PATH):
        print(f"❌ Error: File {CAPTIONS_PATH} does not exist!")
        return
    
    df = pd.read_csv(CAPTIONS_PATH)
    
    if "caption" not in df.columns or "id" not in df.columns:
        print("❌ Error: 'caption' and/or 'id' column not found in captions.csv")
        return
    
    print(f"📊 Processing {len(df)} captions...")
    
    df["cleaned_caption"] = df["caption"].fillna("").apply(clean_text)
    
    features_list = []
    for idx, row in df.iterrows():
        features = extract_text_features(row["cleaned_caption"])
        features["id"] = row["id"]
        features_list.append(features)
    
    text_features = pd.DataFrame(features_list)
    
    cols = ['id'] + [col for col in text_features.columns if col != 'id']
    text_features = text_features[cols]
    
    os.makedirs("features", exist_ok=True)
    
    text_features.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Structural text features extracted and saved to {OUTPUT_PATH}")
    print(f"📈 Extracted {len(text_features)} samples with {len(text_features.columns)-1} structural features each")
    
    generate_tfidf_features(df)

if __name__ == "__main__":
    main()