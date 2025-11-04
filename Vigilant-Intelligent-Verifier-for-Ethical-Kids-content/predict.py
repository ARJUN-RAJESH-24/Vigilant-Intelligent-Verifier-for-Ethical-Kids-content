import pandas as pd
import numpy as np
import joblib
import os
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Feature Extraction Definitions (Replicated from extract_text_features.py) ---
# These functions must be here to generate features during prediction.

# Expanded keyword lists
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

def extract_structural_features(caption):
    """Extract structural and sentiment text features from caption."""
    
    # NOTE: The default return must match the 12 structural features exactly.
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
    
# --- ContentClassifier Class ---

class ContentClassifier:
    
    # Configuration paths for saved models/assets
    TFIDF_MODEL_PATH = "models/tfidf_vectorizer.pkl"
    FEATURE_SCHEMA_PATH = "models/final_feature_schema.txt" # Now points to the definitive text-only schema
    
    def __init__(self, model_path, scaler_path):
        print("\n4️⃣ REAL PREDICTION TESTS")
        print("="*80)
        print(f"✅ Loaded model: {os.path.basename(model_path)}")
        self.model = joblib.load(model_path)
        
        print(f"✅ Loaded scaler: {os.path.basename(scaler_path)}")
        self.scaler = joblib.load(scaler_path)
        
        # Load the fitted TF-IDF Vectorizer
        print(f"✅ Loaded TF-IDF model: {os.path.basename(self.TFIDF_MODEL_PATH)}")
        self.tfidf_vectorizer = joblib.load(self.TFIDF_MODEL_PATH)
        
        # Load the definitive feature schema (512 Text/TFIDF features)
        try:
            with open(self.FEATURE_SCHEMA_PATH, 'r') as f:
                self.feature_names = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"CRITICAL: Schema file not found at {self.FEATURE_SCHEMA_PATH}. Run training first.")

        print(f"✅ Loaded feature schema: {len(self.feature_names)} features")
        
        # CRITICAL CHECK: Ensure the schema only contains text features
        if any(name.startswith(('rms_', 'avg_brightness', 'video_feat')) for name in self.feature_names):
             print(f"❌ ERROR: Schema contains non-text features. Retrain model in text-only mode.")
             
        # NOTE: The model will only accept the features in self.feature_names, automatically ignoring the missing video/audio features.


    def _extract_all_features(self, caption):
        """Extracts and aligns ONLY the Text/TF-IDF features (512 total)."""
        
        # 1. Structural Features (12 features)
        structural_feats = extract_structural_features(caption)
        
        # 2. TF-IDF Features (~500 features)
        cleaned_caption = clean_text(caption)
        tfidf_matrix = self.tfidf_vectorizer.transform([cleaned_caption]).toarray()
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # 3. Combined Feature Construction
        
        # Start with the full list of expected features, initialized to 0.0
        full_features = {name: 0.0 for name in self.feature_names}
        
        # Update the dictionary with extracted structural features
        for k, v in structural_feats.items():
             if k in full_features:
                 full_features[k] = v
        
        # Update the dictionary with extracted TF-IDF features
        for i, col_name in enumerate(tfidf_feature_names):
            if col_name in full_features:
                 full_features[col_name] = tfidf_matrix[0, i]
        
        # Create the final DataFrame, guaranteeing the column order
        X_values = [full_features.get(name, 0.0) for name in self.feature_names]
        X_df = pd.DataFrame([X_values], columns=self.feature_names) 

        return X_df

    def predict(self, caption):
        """Runs the prediction pipeline for a single caption."""
        print("="*80)
        print("🔍 ANALYZING CONTENT")
        print("="*80)
        print(f"Caption: {caption}...")
        
        print("\n📊 Extracting features...")
        X = self._extract_all_features(caption)
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        confidence = probabilities[prediction]
        
        print("="*80)
        print("📊 PREDICTION RESULT")
        print("="*80)
        
        prediction_label = 1 # Adult (Unsafe)
        prediction_text = "🔴 Prediction: Adult"
        
        if prediction == 0:
            prediction_label = 0 # Safe
            prediction_text = "🟢 Prediction: Safe"
        
        print(prediction_text)
        print(f"   Confidence: {confidence:.2%}")
        print("\n   Probabilities:")
        print(f"   - Safe:  {probabilities[0]:.2%}")
        print(f"   - Adult: {probabilities[1]:.2%}")
        
        return {
            'prediction_label': prediction_label,
            'prediction_text': prediction_text,
            'confidence': confidence,
            'probabilities': probabilities
        }

if __name__ == "__main__":
    # NOTE: This block is for manual testing. It requires train_models.py to have run first!
    
    try:
        classifier = ContentClassifier(
            model_path='models/trained_models/RandomForest.pkl',
            scaler_path='models/scaler.pkl'
        )
        
        print("\n--- Manual Test Cases ---")
        
        # Test case 1: Should be Safe (0)
        classifier.predict("Kids playing soccer in the park")
        
        # Test case 2: Should be Adult (1)
        classifier.predict("Sexy model photoshoot behind the scenes")

    except FileNotFoundError as e:
        print(f"\n❌ Error: Cannot run manual test. {e}")