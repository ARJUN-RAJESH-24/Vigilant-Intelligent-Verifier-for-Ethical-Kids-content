import pandas as pd
import os

# --- Configuration ---
NUM_SAMPLES = 300
TEXT_FEATURES_PATH = "features/text_features.csv"
TFIDF_FEATURES_PATH = "features/tfidf_features.csv"
VIDEO_FEATURES_PATH = "features/video_features.csv"
AUDIO_FEATURES_PATH = "features/audio_features.csv"
LABELS_PATH = "data/labels.csv"
TEST_FEATURES_PATH = "features/consolidated_test_features_10_samples.csv"
# --- End Configuration ---

def generate_vid_ids(num):
    return [f"vid{i:03d}" for i in range(1, num + 1)]

def create_test_data():
    print("--- Starting FINAL Feature Consolidation (Text-Only Mode) ---")
    
    try:
        # Load all feature data
        labels_df = pd.read_csv(LABELS_PATH)
        audio_df = pd.read_csv(AUDIO_FEATURES_PATH)
        video_df = pd.read_csv(VIDEO_FEATURES_PATH)
        text_df = pd.read_csv(TEXT_FEATURES_PATH)
        tfidf_df = pd.read_csv(TFIDF_FEATURES_PATH)

        # Apply ID Fixes (necessary to ensure merge works)
        def align_ids(df, length):
            if len(df) >= length:
                df['id'] = generate_vid_ids(len(df))[:len(df)]
            return df

        labels_df = align_ids(labels_df, len(labels_df))
        text_df = align_ids(text_df, len(text_df))
        tfidf_df = align_ids(tfidf_df, len(tfidf_df))
        
        # Inner Merge ONLY Text and TFIDF features (The core training data)
        df = labels_df.merge(text_df, on='id', how='inner')
        df = df.merge(tfidf_df, on='id', how='inner')
        
        # Select the 10 test samples
        test_ids = generate_vid_ids(10)
        test_df = df[df['id'].isin(test_ids)].sort_values(by='id').reset_index(drop=True)
        
        if len(test_df) != 10:
            print(f"\n❌ CRITICAL: Merge resulted in {len(test_df)} rows instead of 10. Test cannot run.")
            return

        test_features_only = test_df.drop(columns=['id', 'label'])
        
        # Save the consolidated file
        os.makedirs(os.path.dirname(TEST_FEATURES_PATH) or 'features', exist_ok=True)
        test_features_only.to_csv(TEST_FEATURES_PATH, index=False)
        
        print(f"\n✅ Final consolidated test features successfully saved to {TEST_FEATURES_PATH}")
        print(f"Dataset shape: {test_features_only.shape}")
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during consolidation: {e}")

if __name__ == "__main__":
    create_test_data()