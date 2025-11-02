import os
import subprocess
import tarfile
import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download

# --- Configuration ---
# Use the Multi-label Classification Dataset, as it contains 12 harm tags (S1: Adult)
DATASET_NAME = "PKU-Alignment/SafeSora-Label"
DOWNLOAD_DIR = "data/safesora_raw"
VIDEO_TARGET_DIR = "data/videos"
CONFIG_FILE = os.path.join(DOWNLOAD_DIR, "config-train.json.gz")
TAR_FILE = os.path.join(DOWNLOAD_DIR, "videos.tar.gz")
# ---------------------

def run_download_command(repo_id, local_dir):
    """Downloads the dataset from Hugging Face using snapshot_download, skipping warnings."""
    print(f"\n📥 Starting download for {repo_id}...\n")

    # The file contains all videos, but we MUST download the config first to know the paths.
    # We explicitly list the files needed for the next step (config files and the video archive).
    
    try:
        # Download dataset with warnings fixed and optimized arguments
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            # Removed deprecated arguments: local_dir_use_symlinks and resume_download
            allow_patterns=["config-train.json.gz", "videos.tar.gz"],
            tqdm_class=tqdm  # enables nice progress bar display
        )
        print(f"\n✅ Download successful for {repo_id}. Files saved in '{local_dir}'.")
        return True
    except Exception as e:
        print(f"❌ Error downloading {repo_id}. Ensure you are authorized for the dataset.")
        print(f"Details: {e}")
        return False


def extract_videos(tar_path, target_dir):
    """Extracts the videos.tar.gz file."""
    print(f"\n📂 Starting extraction of videos from {tar_path}...")

    # We do not modify extraction, as all video files (including S1) are necessary
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            for member in tqdm(members, desc="Extracting videos"):
                tar.extract(member, path=target_dir)
        print(f"✅ Extraction complete. Videos are now in '{target_dir}'.")
        return True
    except Exception as e:
        print(f"❌ Error extracting videos: {e}")
        return False


def main():
    """Main function to download and prepare the SafeSora video data."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(VIDEO_TARGET_DIR, exist_ok=True)

    # --- Step 1: Download the config files and the compressed videos ---
    if not run_download_command(DATASET_NAME, DOWNLOAD_DIR):
        return

    # --- Step 2: Extract the videos ---
    if not os.path.exists(TAR_FILE):
        print(f"❌ CRITICAL: Expected video archive '{TAR_FILE}' not found after download.")
        return

    if not extract_videos(TAR_FILE, VIDEO_TARGET_DIR):
        return

    print("\n" + "=" * 80)
    print("🚀 SAFE-SORA DATA PREPARATION COMPLETE (Raw Files Available)")
    print("=" * 80)
    print(f"You have downloaded: \n- Data config: {CONFIG_FILE} \n- Video files: In {VIDEO_TARGET_DIR}")
    print("\n➡️ Next Step: Run the curator script: python select_safesora_videos.py")


if __name__ == "__main__":
    main()