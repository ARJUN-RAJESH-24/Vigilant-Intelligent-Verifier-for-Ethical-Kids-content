import pandas as pd
from textblob import TextBlob
import re
import os
import numpy as np

CAPTIONS_PATH = "data/captions.csv"
OUTPUT_PATH = "features/text_features.csv"

# Expanded keyword lists for better detection
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
    """
    Clean text by removing URLs, mentions, hashtags, and special characters.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r"@\S+|#\S+", "", text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r"[^A-Za-z0-9\s.,!?]+", "", text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text.lower().strip()

def extract_text_features(caption):
    """
    Extract comprehensive text features from caption.
    
    Features:
    - Basic statistics (char count, word count, avg word length)
    - Sentiment analysis (polarity, subjectivity)
    - Keyword counts (adult, violence)
    - Linguistic features (uppercase ratio, punctuation ratio)
    - Emojis and special characters
    """
    if not caption or len(caption) == 0:
        return {
            "char_count": 0,
            "word_count": 0,
            "avg_word_length": 0,
            "sentiment_polarity": 0,
            "sentiment_subjectivity": 0,
            "adult_keywords_count": 0,
            "violence_keywords_count": 0,
            "uppercase_ratio": 0,
            "punctuation_count": 0,
            "exclamation_count": 0,
            "question_count": 0,
            "number_count": 0,
        }
    
    # Basic statistics
    char_count = len(caption)
    words = caption.split()
    word_count = len(words)
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    
    # Sentiment analysis
    try:
        blob = TextBlob(caption)
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity
    except:
        sentiment_polarity = 0
        sentiment_subjectivity = 0
    
    # Keyword counting
    caption_lower = caption.lower()
    adult_count = sum(1 for keyword in ADULT_KEYWORDS if keyword in caption_lower)
    violence_count = sum(1 for keyword in VIOLENCE_KEYWORDS if keyword in caption_lower)
    
    # Linguistic features
    uppercase_count = sum(1 for c in caption if c.isupper())
    uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
    
    punctuation_count = sum(1 for c in caption if c in ".,;:!?")
    exclamation_count = caption.count("!")
    question_count = caption.count("?")
    number_count = sum(1 for c in caption if c.isdigit())
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": float(avg_word_length),
        "sentiment_polarity": float(sentiment_polarity),
        "sentiment_subjectivity": float(sentiment_subjectivity),
        "adult_keywords_count": adult_count,
        "violence_keywords_count": violence_count,
        "uppercase_ratio": float(uppercase_ratio),
        "punctuation_count": punctuation_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "number_count": number_count,
    }

def main():
    """Main function to extract text features from captions."""
    if not os.path.exists(CAPTIONS_PATH):
        print(f"❌ Error: File {CAPTIONS_PATH} does not exist!")
        return
    
    # Load captions
    df = pd.read_csv(CAPTIONS_PATH)
    
    # Validate required columns
    if "caption" not in df.columns:
        print("❌ Error: 'caption' column not found in captions.csv")
        return
    
    if "id" not in df.columns:
        print("❌ Error: 'id' column not found in captions.csv")
        return
    
    print(f"📊 Processing {len(df)} captions...")
    
    # Clean captions
    df["cleaned_caption"] = df["caption"].fillna("").apply(clean_text)
    
    # Extract features
    features_list = []
    for idx, row in df.iterrows():
        features = extract_text_features(row["cleaned_caption"])
        features["id"] = row["id"]
        features_list.append(features)
    
    # Create features DataFrame
    text_features = pd.DataFrame(features_list)
    
    # Reorder columns to have 'id' first
    cols = ['id'] + [col for col in text_features.columns if col != 'id']
    text_features = text_features[cols]
    
    # Create output directory
    os.makedirs("features", exist_ok=True)
    
    # Save features
    text_features.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Text features extracted and saved to {OUTPUT_PATH}")
    print(f"📈 Extracted {len(text_features)} samples with {len(text_features.columns)-1} features each")

if __name__ == "__main__":
    main()