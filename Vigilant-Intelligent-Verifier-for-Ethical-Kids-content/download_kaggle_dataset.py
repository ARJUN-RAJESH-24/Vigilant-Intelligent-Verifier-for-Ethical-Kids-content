"""
download_kaggle_datasets.py - Download datasets from Kaggle

This script helps download relevant datasets from Kaggle for training.

SETUP:
1. Install kaggle: pip install kaggle
2. Get API credentials from https://www.kaggle.com/settings
3. Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)
"""

import os
import subprocess
import sys

print("="*80)
print("📥 KAGGLE DATASET DOWNLOADER")
print("="*80)

# ============================================================================
# CHECK KAGGLE API
# ============================================================================

print("\n1️⃣ Checking Kaggle API setup...")

try:
    import kaggle
    print("✅ Kaggle package installed")
except ImportError:
    print("❌ Kaggle package not installed")
    print("\nInstalling kaggle...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    import kaggle
    print("✅ Kaggle installed successfully")

# Check for API credentials
kaggle_dir = os.path.expanduser("~/.kaggle")
kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

if not os.path.exists(kaggle_json):
    print("\n❌ Kaggle API credentials not found!")
    print("\n📋 Setup Instructions:")
    print("="*80)
    print("""
1. Go to https://www.kaggle.com/settings
2. Scroll to 'API' section
3. Click 'Create New API Token'
4. This downloads 'kaggle.json'

5. Place it in:
   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
   - Linux/Mac: ~/.kaggle/kaggle.json

6. Run this script again

Note: The .kaggle directory should have restricted permissions (0600 on Linux/Mac)
    """)
    sys.exit(1)

print("✅ Kaggle API credentials found")

# ============================================================================
# RECOMMENDED DATASETS
# ============================================================================

DATASETS = {
    "1": {
        "name": "Hate Speech and Offensive Language",
        "dataset": "mrmorj/hate-speech-and-offensive-language-dataset",
        "size": "~1.5 MB",
        "samples": "~25K tweets",
        "categories": ["hate_speech", "offensive", "safe"]
    },
    "2": {
        "name": "Jigsaw Toxic Comment Classification",
        "dataset": "julian3833/jigsaw-toxic-comment-classification-challenge",
        "size": "~70 MB",
        "samples": "~160K comments",
        "categories": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    },
    "3": {
        "name": "Twitter Hate Speech",
        "dataset": "arkhoshghalb/twitter-sentiment-analysis-hatred-speech",
        "size": "~4 MB",
        "samples": "~30K tweets",
        "categories": ["hate_speech", "safe"]
    },
    "4": {
        "name": "Reddit Mental Health Dataset",
        "dataset": "nikhileswarkomati/suicide-watch",
        "size": "~15 MB",
        "samples": "~230K posts",
        "categories": ["suicide", "depression", "anxiety", "safe"]
    },
    "5": {
        "name": "IMDB Movie Reviews (Sentiment)",
        "dataset": "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        "size": "~30 MB",
        "samples": "~50K reviews",
        "categories": ["positive", "negative"]
    },
    "6": {
        "name": "Twitter Sentiment140",
        "dataset": "kazanova/sentiment140",
        "size": "~80 MB",
        "samples": "~1.6M tweets",
        "categories": ["positive", "negative", "neutral"]
    },
    "7": {
        "name": "YouTube Spam Comments",
        "dataset": "ahsenwaheed/youtube-comments-spam-dataset",
        "size": "~2 MB",
        "samples": "~2K comments",
        "categories": ["spam", "ham"]
    },
    "8": {
        "name": "Sexist Tweets Dataset",
        "dataset": "anilkumar18/sexist-tweet-dataset",
        "size": "~5 MB",
        "samples": "~20K tweets",
        "categories": ["sexist", "not_sexist"]
    }
}

# ============================================================================
# DISPLAY MENU
# ============================================================================

print("\n2️⃣ Available Datasets:")
print("="*80)

for key, info in DATASETS.items():
    print(f"\n[{key}] {info['name']}")
    print(f"    Dataset: {info['dataset']}")
    print(f"    Size: {info['size']}")
    print(f"    Samples: {info['samples']}")
    print(f"    Categories: {', '.join(info['categories'])}")

print("\n[A] Download ALL datasets (recommended)")
print("[Q] Quit")

# ============================================================================
# USER SELECTION
# ============================================================================

print("\n3️⃣ Download Selection:")
print("="*80)

choice = input("\nEnter your choice (1-8, A for all, Q to quit): ").strip().upper()

if choice == 'Q':
    print("Exiting...")
    sys.exit(0)

# Create download directory
download_dir = "data/kaggle_datasets"
os.makedirs(download_dir, exist_ok=True)

# ============================================================================
# DOWNLOAD FUNCTION
# ============================================================================

def download_dataset(dataset_id, dataset_info):
    """Download a single dataset from Kaggle."""
    
    print(f"\n📥 Downloading: {dataset_info['name']}")
    print(f"   Dataset: {dataset_id}")
    
    try:
        # Create subdirectory for this dataset
        dataset_name = dataset_id.split('/')[-1]
        target_dir = os.path.join(download_dir, dataset_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # Download using kaggle API
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        print(f"   Downloading to: {target_dir}")
        api.dataset_download_files(dataset_id, path=target_dir, unzip=True)
        
        print(f"   ✅ Downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

# ============================================================================
# DOWNLOAD DATASETS
# ============================================================================

if choice == 'A':
    print("\n📥 Downloading ALL datasets...")
    success_count = 0
    
    for key, info in DATASETS.items():
        if download_dataset(info['dataset'], info):
            success_count += 1
    
    print(f"\n✅ Successfully downloaded {success_count}/{len(DATASETS)} datasets")

elif choice in DATASETS:
    info = DATASETS[choice]
    download_dataset(info['dataset'], info)

else:
    print("❌ Invalid choice")
    sys.exit(1)

# ============================================================================
# PROCESSING DOWNLOADED DATA
# ============================================================================

print("\n4️⃣ Processing downloaded datasets...")
print("="*80)

processed_samples = []

# Process each downloaded dataset
for dataset_name in os.listdir(download_dir):
    dataset_path = os.path.join(download_dir, dataset_name)
    
    if os.path.isdir(dataset_path):
        print(f"\n📂 Processing {dataset_name}...")
        
        # Find CSV files
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            csv_path = os.path.join(dataset_path, csv_file)
            
            try:
                import pandas as pd
                df = pd.read_csv(csv_path, nrows=1000)  # Process first 1000 rows
                
                print(f"   ✅ Loaded {csv_file}: {df.shape}")
                print(f"      Columns: {list(df.columns)[:5]}...")
                
                # Find text and label columns
                text_cols = [col for col in df.columns if any(
                    x in col.lower() for x in ['text', 'comment', 'tweet', 'content', 'caption']
                )]
                
                label_cols = [col for col in df.columns if any(
                    x in col.lower() for x in ['label', 'class', 'toxic', 'hate', 'target', 'sentiment']
                )]
                
                if text_cols and label_cols:
                    text_col = text_cols[0]
                    label_col = label_cols[0]
                    
                    print(f"      Text column: {text_col}")
                    print(f"      Label column: {label_col}")
                    
                    # Sample extraction
                    for idx, row in df.head(100).iterrows():
                        if pd.notna(row[text_col]):
                            text = str(row[text_col])[:500]
                            label_val = row[label_col]
                            
                            # Convert label to binary (0/1)
                            label = 0
                            if isinstance(label_val, str):
                                if any(x in label_val.lower() for x in ['hate', 'toxic', 'offensive', 'negative', '1']):
                                    label = 1
                            elif label_val > 0:
                                label = 1
                            
                            processed_samples.append({
                                'id': f'{dataset_name}_{idx:05d}',
                                'caption': text,
                                'label': label,
                                'source': dataset_name
                            })
                
            except Exception as e:
                print(f"   ⚠️  Error processing {csv_file}: {e}")

# Save processed data
if processed_samples:
    df_processed = pd.DataFrame(processed_samples)
    
    print(f"\n✅ Processed {len(df_processed)} samples from Kaggle datasets")
    print(f"   Label distribution:")
    print(df_processed['label'].value_counts())
    
    # Save to CSV
    output_file = "data/kaggle_processed.csv"
    df_processed.to_csv(output_file, index=False)
    print(f"\n✅ Saved to {output_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ KAGGLE DOWNLOAD COMPLETE!")
print("="*80)

print(f"""
📁 Downloaded to: {download_dir}

📊 Processed samples: {len(processed_samples) if processed_samples else 0}

➡️  Next steps:
   1. Run data collection: python collect_datasets.py
   2. This will merge Kaggle data with synthetic data
   3. Then train models: python scripts/train_models.py

💡 Tip: You can manually explore the downloaded datasets in:
    {download_dir}
""")