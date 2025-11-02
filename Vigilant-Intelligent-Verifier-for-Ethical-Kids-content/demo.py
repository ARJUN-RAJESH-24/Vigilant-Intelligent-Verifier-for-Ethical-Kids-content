"""
demo.py - Complete demonstration of the adult content classifier

This script demonstrates:
1. Creating sample data
2. Extracting features
3. Training models
4. Making predictions
5. Evaluating results
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("🎬 ADULT CONTENT CLASSIFIER - COMPLETE DEMO")
print("="*80)

# ============================================================================
# STEP 1: CREATE SAMPLE DATA
# ============================================================================
print("\n" + "="*80)
print("1️⃣ CREATING SAMPLE DATA")
print("="*80)

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("data/videos", exist_ok=True)
os.makedirs("features", exist_ok=True)
os.makedirs("models/trained_models", exist_ok=True)
os.makedirs("models/results", exist_ok=True)

# Create sample captions with diverse content
sample_captions = pd.DataFrame({
    "id": [
        "sample001", "sample002", "sample003", "sample004", "sample005",
        "sample006", "sample007", "sample008", "sample009", "sample010",
        "sample011", "sample012", "sample013", "sample014", "sample015",
        "sample016", "sample017", "sample018", "sample019", "sample020"
    ],
    "caption": [
        # Safe content (label 0)
        "Family vacation at the beach with kids playing in sand",
        "Cooking tutorial: How to make healthy pasta at home",
        "Morning yoga and meditation for beginners",
        "Educational science experiment with colorful chemicals",
        "Travel vlog: Exploring ancient temples in Asia",
        "DIY home renovation project step by step guide",
        "Cute puppies playing in the garden compilation",
        "Gaming walkthrough final boss battle strategy",
        "Professional makeup tutorial for everyday look",
        "Nature documentary about wildlife in Africa",
        
        # Adult content (label 1)
        "Hot dance performance at nightclub with DJ music",
        "Romantic couple intimate kiss scene from movie",
        "Sexy model photoshoot behind the scenes footage",
        "Late night party with provocative dancing",
        "Bedroom scene with sensual lighting and music",
        "Adult themed comedy show explicit content warning",
        "Lingerie fashion show runway models backstage",
        "Mature audience only erotic art photography",
        "Nude figure drawing class for art students",
        "NSFW content warning provocative music video"
    ]
})

# Create labels (0 = safe, 1 = adult)
sample_labels = pd.DataFrame({
    "id": [
        "sample001", "sample002", "sample003", "sample004", "sample005",
        "sample006", "sample007", "sample008", "sample009", "sample010",
        "sample011", "sample012", "sample013", "sample014", "sample015",
        "sample016", "sample017", "sample018", "sample019", "sample020"
    ],
    "label": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Safe
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1   # Adult
    ]
})

# Save to CSV
sample_captions.to_csv("data/captions.csv", index=False)
sample_labels.to_csv("data/labels.csv", index=False)

print(f"✅ Created {len(sample_captions)} sample captions")
print(f"✅ Created {len(sample_labels)} sample labels")
print(f"   - Safe content: {(sample_labels['label'] == 0).sum()}")
print(f"   - Adult content: {(sample_labels['label'] == 1).sum()}")

# ============================================================================
# STEP 2: EXTRACT TEXT FEATURES
# ============================================================================
print("\n" + "="*80)
print("2️⃣ EXTRACTING TEXT FEATURES")
print("="*80)

sys.path.append('scripts')
from extract_text_features import extract_text_features, clean_text

# Extract text features
text_features_list = []
for idx, row in sample_captions.iterrows():
    cleaned = clean_text(row['caption'])
    features = extract_text_features(cleaned)
    features['id'] = row['id']
    text_features_list.append(features)

text_df = pd.DataFrame(text_features_list)
text_df.to_csv("features/text_features.csv", index=False)

print(f"✅ Extracted {len(text_df)} text feature sets")
print(f"   Features per sample: {len(text_df.columns) - 1}")
print(f"\n📊 Sample features:")
print(text_df[['id', 'char_count', 'word_count', 'adult_keywords_count', 'sentiment_polarity']].head())

# ============================================================================
# STEP 3: CREATE SYNTHETIC AUDIO/VIDEO FEATURES (FOR DEMO)
# ============================================================================
print("\n" + "="*80)
print("3️⃣ CREATING SYNTHETIC AUDIO/VIDEO FEATURES (Demo Mode)")
print("="*80)
print("⚠️  Note: In production, use real videos and run extract_audio/video_features.py")

# Create synthetic audio features (realistic ranges)
np.random.seed(42)
audio_features = []
for sample_id in sample_captions['id']:
    label = sample_labels[sample_labels['id'] == sample_id]['label'].values[0]
    
    # Adult content tends to have different audio characteristics
    if label == 1:
        # Higher energy, more bass
        rms_mean = np.random.uniform(0.05, 0.15)
        tempo = np.random.uniform(120, 140)
    else:
        # Lower energy, varied tempo
        rms_mean = np.random.uniform(0.02, 0.08)
        tempo = np.random.uniform(80, 120)
    
    features = {
        'id': sample_id,
        'rms_mean': rms_mean,
        'rms_std': np.random.uniform(0.01, 0.03),
        'zcr_mean': np.random.uniform(0.05, 0.15),
        'zcr_std': np.random.uniform(0.01, 0.05),
        'spectral_centroid_mean': np.random.uniform(1000, 3000),
        'spectral_centroid_std': np.random.uniform(200, 500),
        'spectral_bandwidth_mean': np.random.uniform(1500, 2500),
        'spectral_bandwidth_std': np.random.uniform(300, 600),
        'spectral_rolloff_mean': np.random.uniform(2000, 4000),
        'spectral_rolloff_std': np.random.uniform(400, 800),
        'tempo': tempo,
        'chroma_mean': np.random.uniform(0.3, 0.7),
        'chroma_std': np.random.uniform(0.1, 0.3),
    }
    
    # Add MFCC features
    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.random.uniform(-50, 50)
        features[f'mfcc_{i}_std'] = np.random.uniform(5, 20)
    
    audio_features.append(features)

audio_df = pd.DataFrame(audio_features)
audio_df.to_csv("features/audio_features.csv", index=False)

print(f"✅ Created {len(audio_df)} synthetic audio feature sets")
print(f"   Features per sample: {len(audio_df.columns) - 1}")

# Create synthetic video features
video_features = []
for sample_id in sample_captions['id']:
    label = sample_labels[sample_labels['id'] == sample_id]['label'].values[0]
    
    # Adult content tends to have different visual characteristics
    if label == 1:
        # Higher skin ratio, different brightness
        skin_ratio = np.random.uniform(0.15, 0.35)
        brightness = np.random.uniform(100, 150)
        motion = np.random.uniform(20, 40)
    else:
        # Lower skin ratio, varied brightness
        skin_ratio = np.random.uniform(0.05, 0.15)
        brightness = np.random.uniform(80, 140)
        motion = np.random.uniform(10, 30)
    
    features = {
        'id': sample_id,
        'frame_count': np.random.randint(200, 500),
        'processed_frames': np.random.randint(40, 100),
        'fps': 30.0,
        'duration_seconds': np.random.uniform(7, 15),
        'brightness_mean': brightness,
        'brightness_std': np.random.uniform(10, 30),
        'brightness_min': brightness - np.random.uniform(20, 40),
        'brightness_max': brightness + np.random.uniform(20, 40),
        'motion_mean': motion,
        'motion_std': np.random.uniform(5, 15),
        'motion_max': motion + np.random.uniform(10, 30),
        'skin_ratio_mean': skin_ratio,
        'skin_ratio_std': np.random.uniform(0.02, 0.08),
        'skin_ratio_max': skin_ratio + np.random.uniform(0.05, 0.15),
        'edge_density_mean': np.random.uniform(0.1, 0.3),
        'edge_density_std': np.random.uniform(0.02, 0.08),
        'hue_mean': np.random.uniform(50, 150),
        'hue_std': np.random.uniform(20, 50),
        'saturation_mean': np.random.uniform(50, 150),
        'saturation_std': np.random.uniform(20, 50),
        'scene_cuts': np.random.randint(2, 10),
        'scene_cut_rate': np.random.uniform(0.02, 0.10),
    }
    video_features.append(features)

video_df = pd.DataFrame(video_features)
video_df.to_csv("features/video_features.csv", index=False)

print(f"✅ Created {len(video_df)} synthetic video feature sets")
print(f"   Features per sample: {len(video_df.columns) - 1}")

# ============================================================================
# STEP 4: TRAIN MODELS
# ============================================================================
print("\n" + "="*80)
print("4️⃣ TRAINING MACHINE LEARNING MODELS")
print("="*80)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# Merge all features
df = text_df.merge(audio_df, on='id').merge(video_df, on='id').merge(sample_labels, on='id')
df = df.fillna(0)

X = df.drop(columns=['id', 'label'])
y = df['label']

print(f"📊 Dataset shape: {X.shape}")
print(f"   Total features: {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
print(f"\n✅ Scaler saved")

# Train multiple models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

results = []
for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"   ✅ Accuracy: {accuracy:.4f}")
    print(f"   ✅ F1-Score: {f1:.4f}")
    
    # Save model
    joblib.dump(model, f"models/trained_models/{name}.pkl")
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'F1_Score': f1
    })

results_df = pd.DataFrame(results).sort_values('F1_Score', ascending=False)
print(f"\n📊 Model Comparison:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name}")

# ============================================================================
# STEP 5: MAKE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("5️⃣ MAKING PREDICTIONS ON NEW CONTENT")
print("="*80)

# Load best model
best_model = joblib.load(f"models/trained_models/{best_model_name}.pkl")

# Test predictions
test_captions = [
    "Educational cooking show for family dinner",
    "Hot sexy dance club night performance",
    "Kids playing soccer in the park"
]

print("\n🔍 Testing predictions on new captions:")
for caption in test_captions:
    # Extract features
    cleaned = clean_text(caption)
    text_feats = extract_text_features(cleaned)
    
    # Create dummy audio/video features (in production, extract from real content)
    all_features = {**text_feats}
    for col in X.columns:
        if col not in all_features:
            all_features[col] = 0
    
    # Create feature vector in correct order
    feature_vector = [all_features[col] for col in X.columns]
    feature_vector_scaled = scaler.transform([feature_vector])
    
    # Predict
    prediction = best_model.predict(feature_vector_scaled)[0]
    proba = best_model.predict_proba(feature_vector_scaled)[0]
    
    result = "🔴 ADULT" if prediction == 1 else "🟢 SAFE"
    confidence = proba[prediction]
    
    print(f"\n   Caption: \"{caption}\"")
    print(f"   {result} (Confidence: {confidence:.2%})")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("✅ DEMO COMPLETED SUCCESSFULLY!")
print("="*80)

print("""
📁 Files Created:
   ✓ data/captions.csv
   ✓ data/labels.csv
   ✓ features/text_features.csv
   ✓ features/audio_features.csv
   ✓ features/video_features.csv
   ✓ models/scaler.pkl
   ✓ models/trained_models/*.pkl

📊 Results:
   ✓ Trained 4 ML models
   ✓ Best model: {best_model_name}
   ✓ Ready for predictions

🚀 Next Steps:
   1. Use real videos: Add to data/videos/
   2. Extract real features: Run extract_*_features.py scripts
   3. Retrain with more data for better accuracy
   4. Use predict.py for production predictions

💡 For production use:
   python predict.py --video path/to/video.mp4 --caption "Description"
""".format(best_model_name=best_model_name))

print("="*80)