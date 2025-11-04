import pandas as pd
import os
import shutil
import json
import gzip
from tqdm import tqdm

# --- Configuration ---
RAW_CONFIG_PATH = "data/safesora_raw/config-train.json.gz"
# RAW_VIDEO_DIR is data/videos. The script assumes the raw extracted "videos/" folder 
# (which contains the nested files) is placed here.
RAW_VIDEO_DIR = "data/videos" 
FINAL_VIDEO_DIR = "data/videos" # Target for the clean vid001.mp4 files
FINAL_LABELS_PATH = "data/labels.csv"
# ---------------------

# SafeSora Harm Tags: S1 is Adult Content
ADULT_HARMS_KEY = "S1: Adult, Explicit Sexual Content"
NUM_SAMPLES_PER_CLASS = 150
TARGET_TOTAL_SAMPLES = NUM_SAMPLES_PER_CLASS * 2

def load_and_normalize_config(config_path):
    """
    Loads and flattens the SafeSora JSON.GZ config file, robustly skipping bad lines.
    (FIXED: Implemented robust JSONL parsing to skip corrupted lines.)
    """
    print(f"📂 Loading and normalizing config from {config_path}...")
    
    records = []
    skipped_lines = 0

    try:
        with gzip.open(config_path, 'rt', encoding='utf8') as f:
            for line in tqdm(f, desc="Reading JSON lines"):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped_lines += 1
    except FileNotFoundError:
        print(f"❌ ERROR: Config file not found at {config_path}. Did you run download_safesora_data.py?")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ ERROR accessing config file: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    
    if skipped_lines > 0:
        print(f"⚠️ WARNING: Skipped {skipped_lines} non-JSON lines during reading.")

    # Normalize the complex harm labels column
    if 'harm_labels' in df.columns:
        # We need to ensure the nested harm dictionary exists before accessing it
        def get_is_adult(harm_dict):
            return harm_dict.get(ADULT_HARMS_KEY, False) if isinstance(harm_dict, dict) else False

        df['is_adult'] = df['harm_labels'].apply(get_is_adult)
        df = df.drop('harm_labels', axis=1)
    else:
        print("⚠️ WARNING: 'harm_labels' column not found in config. Cannot filter by S1.")
        return pd.DataFrame()
    
    print(f"✅ Loaded {len(df)} records.")
    return df

def select_and_copy_videos(df):
    """
    Selects a balanced set of videos (S1 Adult vs. Safe), copies them, and generates new labels.
    (FIXED: Source path construction handles nested SafeSora directories.)
    """
    
    if df.empty or 'video_path' not in df.columns:
        print("❌ ERROR: Config DataFrame is empty or missing 'video_path'.")
        return

    # --- 1. Define Filter Criteria ---
    
    # 1.1 Unsafe Filter: Select videos explicitly labeled as S1 (Adult/Explicit)
    unsafe_df = df[df.get('is_adult', False) == True].copy()
    
    # 1.2 Safe Filter: Select videos that are NOT explicitly labeled as S1
    safe_df = df[df.get('is_adult', False) == False].copy()
    
    # --- 2. Sample Selection (150 of each) ---
    
    # Drop duplicates and shuffle the data
    unsafe_df = unsafe_df.drop_duplicates(subset=['video_path']).sample(frac=1, random_state=42)
    safe_df = safe_df.drop_duplicates(subset=['video_path']).sample(frac=1, random_state=42)
    
    # Select the required number of samples
    unsafe_samples = unsafe_df.head(NUM_SAMPLES_PER_CLASS)
    safe_samples = safe_df.head(NUM_SAMPLES_PER_CLASS)

    if len(unsafe_samples) < NUM_SAMPLES_PER_CLASS or len(safe_samples) < NUM_SAMPLES_PER_CLASS:
        print(f"⚠️ WARNING: Could only find {len(safe_samples)} safe and {len(unsafe_samples)} adult samples. Proceeding with available count.")
    
    # Combine the selections and shuffle the final list
    final_selection = pd.concat([safe_samples, unsafe_samples]).reset_index(drop=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Final selection: {len(final_selection)} videos ({len(safe_samples)} safe, {len(unsafe_samples)} adult).")

    # --- 3. Copy, Rename, and Generate New Labels ---
    
    new_labels = []
    
    for i, row in tqdm(final_selection.iterrows(), total=len(final_selection), desc="Curating and Copying"):
        target_id = f"vid{(i + 1):03d}"
        label = 1 if row['is_adult'] else 0
        
        # Source path construction: The config 'video_path' provides the nested path 
        # (e.g., videos/prompt_id/video_id.mp4) relative to the extraction base.
        source_path = os.path.join(FINAL_VIDEO_DIR, row['video_path'])
        
        # Determine the correct file extension
        extension = os.path.splitext(row['video_path'])[1]
        
        # Destination path: Clean up the main data/videos folder
        dest_path = os.path.join(FINAL_VIDEO_DIR, f"{target_id}{extension}")

        # Copy the video file
        try:
            if os.path.exists(source_path):
                 shutil.copyfile(source_path, dest_path)
                 new_labels.append({'id': target_id, 'label': label})
            else:
                 print(f"\n❌ FILE NOT FOUND: Could not find source video at {source_path}. Skipping.")
        
        except Exception as e:
            print(f"\n❌ ERROR copying {target_id}: {e}")

    # --- 4. Save New Labels ---
    new_labels_df = pd.DataFrame(new_labels)
    new_labels_df.to_csv(FINAL_LABELS_PATH, index=False)
    
    print(f"\n🎉 Successfully curated and copied {len(new_labels)} videos to {FINAL_VIDEO_DIR}/.")
    print(f"🎉 Updated ground truth labels saved to {FINAL_LABELS_PATH}.")

def main():
    if os.path.exists(RAW_CONFIG_PATH):
        df_config = load_and_normalize_config(RAW_CONFIG_PATH)
        if not df_config.empty:
            select_and_copy_videos(df_config)
    else:
        print("❌ CRITICAL ERROR: SafeSora config file is missing. Please run download_safesora_data.py first.")

if __name__ == "__main__":
    main()