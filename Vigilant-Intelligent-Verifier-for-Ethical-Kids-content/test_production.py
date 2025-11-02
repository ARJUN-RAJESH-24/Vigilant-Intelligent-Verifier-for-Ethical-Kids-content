"""
test_production.py - Production-ready testing with real predictions

This script tests the entire pipeline with comprehensive scenarios.
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

print("="*80)
print("🧪 PRODUCTION TESTING - COMPREHENSIVE VALIDATION")
print("="*80)

# ============================================================================
# TEST 1: Check Data Quality
# ============================================================================
print("\n1️⃣ DATA QUALITY CHECK")
print("="*80)

# Load data
try:
    captions_df = pd.read_csv('data/captions.csv')
    labels_df = pd.read_csv('data/labels.csv')
    # Load all feature files for the check (assuming extract_text_features has run)
    text_features = pd.read_csv('features/text_features.csv') 

    print(f"✅ Captions: {len(captions_df)} samples")
    print(f"✅ Labels: {len(labels_df)} samples")
    print(f"✅ Text features: {len(text_features)} samples")

    # Check for issues
    print("\n🔍 Data Quality Checks:")

    # Check for duplicates
    dup_captions = captions_df['caption'].duplicated().sum()
    print(f"   Duplicate captions: {dup_captions}")

    # Check for missing values
    missing_captions = captions_df['caption'].isna().sum()
    missing_labels = labels_df['label'].isna().sum()
    print(f"   Missing captions: {missing_captions}")
    print(f"   Missing labels: {missing_labels}")

    # Check label distribution
    label_dist = labels_df['label'].value_counts()
    print(f"\n   Label distribution:")
    print(f"   - Safe (0): {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(labels_df)*100:.1f}%)")
    print(f"   - Adult (1): {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(labels_df)*100:.1f}%)")

    # Check balance
    if len(labels_df) > 0 and len(label_dist) == 2:
        
        # --- FIX: Access numpy array values without parentheses ---
        balance_ratio = min(label_dist.values) / max(label_dist.values)
        # --- END FIX ---
        
        if balance_ratio < 0.5:
            print(f"   ⚠️  Dataset imbalanced (ratio: {balance_ratio:.2f})")
        else:
            print(f"   ✅ Dataset balanced (ratio: {balance_ratio:.2f})")
    elif len(labels_df) > 0 and len(label_dist) < 2:
         print("   ⚠️  Cannot check balance: Only one label class found.")
    else:
        print("   ⚠️  Cannot check balance: Empty dataset.")

except Exception as e:
    print(f"❌ Error loading data: {e}")

# ============================================================================
# TEST 2: Feature Statistics
# ============================================================================
print("\n2️⃣ FEATURE STATISTICS")
print("="*80)

try:
    # Exclude 'id' column for describe()
    features_only = text_features.drop(columns=['id'], errors='ignore') 
    print(f"\nText Features ({len(features_only.columns)} features):")
    print(features_only.describe())

    # Check for zero variance features
    zero_var = features_only.select_dtypes(include=[np.number]).columns[
        features_only.select_dtypes(include=[np.number]).std() == 0
    ]
    if len(zero_var) > 0:
        print(f"\n⚠️  Zero variance features: {list(zero_var)}")
    else:
        print(f"\n✅ All features have variance")
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================================================
# TEST 3: Model Performance Check
# ============================================================================
print("\n3️⃣ MODEL PERFORMANCE CHECK")
print("="*80)

if os.path.exists('models/results/model_comparison.csv'):
    results = pd.read_csv('models/results/model_comparison.csv')
    print("\n📊 Model Performance Summary:")
    print(results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    
    # Check for overfitting
    print("\n🔍 Overfitting Check:")
    for idx, row in results.iterrows():
        if pd.notna(row.get('CV_F1_Mean')):
            test_f1 = row['F1-Score']
            cv_f1 = row['CV_F1_Mean']
            gap = test_f1 - cv_f1
            
            status = "✅" if abs(gap) < 0.15 else "⚠️"
            print(f"   {status} {row['Model']}: Test F1={test_f1:.3f}, CV F1={cv_f1:.3f}, Gap={gap:.3f}")
else:
    print("⚠️  No model results found. Run training first.")

# ============================================================================
# TEST 4: Real Prediction Tests
# ============================================================================
print("\n4️⃣ REAL PREDICTION TESTS")
print("="*80)

if os.path.exists('models/trained_models/RandomForest.pkl'):
    try:
        # NOTE: You need to implement ContentClassifier in a 'predict.py' file 
        # that uses the NEW TFIDF_MODEL_PATH and loads TF-IDF features as well!
        from predict import ContentClassifier
        
        classifier = ContentClassifier(
            model_path='models/trained_models/RandomForest.pkl',
            scaler_path='models/scaler.pkl'
        )
        
        # Test cases (The misclassified case is at the top)
        test_cases = [
            ("Family vacation at the beach with children playing", 0, "Safe"),
            ("Cooking tutorial for healthy dinner recipes", 0, "Safe"),
            ("Educational science experiment for students", 0, "Safe"),
            ("Hot sexy dance performance at nightclub", 1, "Adult"),
            ("NSFW explicit adult content warning", 1, "Adult"),
            ("Intimate bedroom scene romantic couples", 1, "Adult"),
            ("Kids playing soccer in the park", 0, "Safe"),
            ("Nude figure drawing art class", 1, "Adult"),
            ("Morning yoga and meditation routine", 0, "Safe"),
            ("Strip club performance late night", 1, "Adult"),
        ]
        
        print("\n🧪 Testing predictions on known cases:")
        print("-" * 80)
        
        correct = 0
        total = len(test_cases)
        
        for caption, expected, category in test_cases:
            result = classifier.predict(caption=caption.strip()) 
            predicted = result['prediction_label']
            confidence = result.get('confidence', 0)
            
            is_correct = (predicted == expected)
            correct += is_correct
            
            status = "✅" if is_correct else "❌"
            emoji = "🟢" if expected == 0 else "🔴"
            
            print(f"{status} {emoji} [{category}] Pred:{predicted}, Expected:{expected}, Conf:{confidence:.2%}")
            print(f"   \"{caption[:60]}...\"")
        
        accuracy = correct / total
        print("-" * 80)
        print(f"\n📊 Prediction Accuracy: {correct}/{total} = {accuracy:.1%}")
        
        if accuracy >= 0.8:
            print("✅ Model performing well on test cases!")
        elif accuracy >= 0.6:
            print("⚠️  Model needs improvement (60-80% accuracy)")
        else:
            print("❌ Model needs retraining (< 60% accuracy)")
    except Exception as e:
        print(f"❌ Error testing predictions: {e}")
else:
    print("⚠️  No trained models found. Run training first.")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 PRODUCTION TEST SUMMARY")
print("="*80)

print("""
✅ Tests Complete! (Next run will reflect TF-IDF improvements)

📖 Recommendations:
    1. **Execute New Pipeline:** Run the feature extraction and training to use TF-IDF features.
    
    2. Add real videos to data/videos/
       → Extract audio/video features
""")

print("="*80)