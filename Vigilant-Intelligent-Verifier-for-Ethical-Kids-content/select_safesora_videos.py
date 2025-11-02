import pandas as pd
import os
import shutil
import json
import gzip
from tqdm import tqdm

# --- Configuration ---
RAW_CONFIG_PATH = "data/safesora_raw/config-train.json.gz"
RAW_VIDEO_DIR = "data/videos"  # Location where videos.tar.gz extracted its files (e.g., data/videos/prompt_id/video_id.mp4)
FINAL_VIDEO_DIR = "data/videos" # The destination folder, already exists
FINAL_LABELS_PATH = "data/labels.csv"
# ---------------------

# SafeSora Harm Tags: S1 is Adult Content
ADULT_HARMS_KEY = "S1: Adult, Explicit Sexual Content"
NUM_SAMPLES_PER_CLASS = 150
TARGET_TOTAL_SAMPLES = NUM_SAMPLES_PER_CLASS * 2

def load_and_normalize_config(config_path):
    """Loads and flattens the SafeSora JSON.GZ config file."""
    print(f"📂 Loading and normalizing config from {config_path}...")
    
    # Read the compressed JSON lines file
    records = []
    try:
        with gzip.open(config_path, 'rt', encoding='utf8') as f:
            for line in tqdm(f, desc="Reading JSON lines"):
                records.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ ERROR: Config file not found at {config_path}. Did you run download_safesora_data.py?")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    
    # Normalize the complex harm labels column (Assuming 'harm_labels' is the key)
    # The 'harm_labels' column contains a dictionary of {Harm_Tag: Boolean}
    if 'harm_labels' in df.columns:
        harm_labels_df = pd.json_normalize(df['harm_labels'])
        # Rename S1 tag for easier use
        if ADULT_HARMS_KEY in harm_labels_df.columns:
             harm_labels_df = harm_labels_df.rename(columns={ADULT_HARMS_KEY: 'is_adult'})
        
        # Merge the harm labels back into the main DataFrame
        df = pd.concat([df.drop('harm_labels', axis=1), harm_labels_df], axis=1)

    print(f"✅ Loaded {len(df)} records.")
    return df

def select_and_copy_videos(df):
    """Selects a balanced set of videos, copies them, and generates new labels."""
    
    if df.empty or 'video_path' not in df.columns:
        print("❌ ERROR: Config DataFrame is empty or missing 'video_path'.")
        return

    # --- 1. Define Filter Criteria ---
    
    # 1.1 Unsafe Filter: Select videos explicitly labeled as Adult/Explicit
    unsafe_df = df[df.get('is_adult', False) == True].copy()
    
    # 1.2 Safe Filter: Select videos where NO harm tags were true (robust safety neutral)
    # We must ensure all other S-tags are False/missing. This depends on the full tag list.
    # For simplicity, we assume 'is_adult' is the main risk and use the rest as potential safe, 
    # but strictly filter to the opposite of adult.
    safe_df = df[df.get('is_adult', False) == False].copy()
    
    # --- 2. Sample Selection (150 of each) ---
    
    # Drop duplicates based on the actual video file path
    unsafe_df = unsafe_df.drop_duplicates(subset=['video_path'])
    safe_df = safe_df.drop_duplicates(subset=['video_path'])
    
    # Select the required number of samples
    if len(unsafe_df) < NUM_SAMPLES_PER_CLASS or len(safe_df) < NUM_SAMPLES_PER_CLASS:
        print(f"⚠️ WARNING: Could only find {len(safe_df)} safe and {len(unsafe_df)} unsafe samples.")
    
    unsafe_samples = unsafe_df.head(NUM_SAMPLES_PER_CLASS)
    safe_samples = safe_df.head(NUM_SAMPLES_PER_CLASS)
    
    final_selection = pd.concat([safe_samples, unsafe_samples]).reset_index(drop=True)
    
    print(f"✅ Final selection: {len(final_selection)} videos ({len(safe_samples)} safe, {len(unsafe_samples)} unsafe).")

    # --- 3. Copy, Rename, and Generate New Labels ---
    
    new_labels = []
    
    for i, row in tqdm(final_selection.iterrows(), total=len(final_selection), desc="Copying and Renaming"):
        # The target ID runs from vid001 to vid300
        target_id = f"vid{(i + 1):03d}"
        
        # Determine the binary label (0 for first 150 (safe), 1 for second 150 (unsafe))
        label = 0 if i < NUM_SAMPLES_PER_CLASS else 1
        
        # Source path construction: video_path is relative to RAW_VIDEO_DIR
        source_path = os.path.join(RAW_VIDEO_DIR, row['video_path'])
        
        # Destination path construction: target_id + .mp4 extension
        # We assume the video file extension is mp4 for simplicity
        dest_path = os.path.join(FINAL_VIDEO_DIR, f"{target_id}.mp4")

        # Copy the video file
        try:
            # We move the video from the nested folder to the top level data/videos/ folder
            shutil.copyfile(source_path, dest_path)
            
            new_labels.append({'id': target_id, 'label': label})
        
        except FileNotFoundError:
            print(f"\n❌ FILE NOT FOUND: Could not find source video at {source_path}. Skipping.")
        except Exception as e:
            print(f"\n❌ ERROR copying {target_id}: {e}")

    # --- 4. Save New Labels ---
    new_labels_df = pd.DataFrame(new_labels)
    new_labels_df.to_csv(FINAL_LABELS_PATH, index=False)
    
    print(f"\n🎉 Successfully copied {len(new_labels)} videos to {FINAL_VIDEO_DIR}/.")
    print(f"🎉 Updated ground truth labels saved to {FINAL_LABELS_PATH}.")
    print("\n⚠️ IMPORTANT: You must delete the intermediate files from data/safesora_raw and data/videos to avoid using duplicate/incorrect files.")

def main():
    if os.path.exists(RAW_CONFIG_PATH):
        df_config = load_and_normalize_config(RAW_CONFIG_PATH)
        if not df_config.empty:
            select_and_copy_videos(df_config)
    else:
        print("❌ CRITICAL ERROR: SafeSora config file is missing. Please run download_safesora_data.py first.")

if __name__ == "__main__":
    main()