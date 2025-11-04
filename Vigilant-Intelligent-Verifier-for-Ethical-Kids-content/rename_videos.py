import os
from tqdm import tqdm
import glob

VIDEO_DIR = "data/videos"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.webm', '.mkv')
TARGET_COUNT = 300

def cleanup_old_vids():
    """Deletes any existing vidXXX.mp4 files in the directory to prevent WinError 183."""
    print(f"🧹 Cleaning up existing vidXXX files in {VIDEO_DIR}...")
    
    deleted_count = 0
    # Find files named vid001.mp4, vid002.mp4, etc.
    for i in range(1, TARGET_COUNT + 1):
        target_name = f"vid{i:03d}"
        
        # Check all common extensions
        for ext in VIDEO_EXTENSIONS:
            file_path = os.path.join(VIDEO_DIR, target_name + ext)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Error deleting old file {file_path}: {e}")
                    
    print(f"✅ Cleaned up {deleted_count} conflicting files.")


def rename_files_sequentially():
    """
    Finds the first 300 video files in the root data/videos directory 
    and renames them to vid001.mp4 through vid300.mp4.
    """
    
    if not os.path.isdir(VIDEO_DIR):
        print(f"❌ ERROR: Video directory not found: {VIDEO_DIR}")
        return

    # --- STEP 1: CLEAN UP CONFLICTS ---
    cleanup_old_vids()

    print(f"\n🎬 Starting renaming process in {VIDEO_DIR}...")
    
    # 2. Find all non-vidXXX video files (which should be the original hex names)
    file_list = []
    
    # We find ALL files, then filter out the target names
    all_files_in_dir = os.listdir(VIDEO_DIR)
    
    for name in all_files_in_dir:
        # Check if it's a video file and NOT already named vidXXX
        if name.lower().endswith(VIDEO_EXTENSIONS) and not name.lower().startswith('vid'):
             file_list.append(os.path.join(VIDEO_DIR, name))

    if not file_list:
        print("❌ ERROR: No hexadecimal video files found to rename. Ensure 'flatten_videos.py' ran correctly.")
        return

    # 3. Rename the first TARGET_COUNT files
    files_renamed = 0
    
    for i in tqdm(range(min(len(file_list), TARGET_COUNT)), desc="Renaming files"):
        old_path = file_list[i]
        
        # Get the original file extension (e.g., .mp4, .avi)
        _, ext = os.path.splitext(old_path)
        
        # Determine the new sequential ID (vid001, vid002, etc.)
        new_id = f"vid{(i + 1):03d}"
        new_path = os.path.join(VIDEO_DIR, f"{new_id}{ext}")
        
        # Rename the file (this time it won't conflict)
        try:
            os.rename(old_path, new_path)
            files_renamed += 1
        except Exception as e:
            # If it fails here, it's a deeper OS error.
            print(f"\n❌ Final Error renaming file {os.path.basename(old_path)}: {e}")
            
    print("\n" + "="*50)
    print(f"✅ Renaming complete! {files_renamed} files renamed.")
    
    if files_renamed == TARGET_COUNT:
        print("🎉 Dataset ready for feature extraction!")
    else:
        print(f"⚠️ Note: Only {files_renamed} files were renamed. Proceeding with available features.")


if __name__ == "__main__":
    rename_files_sequentially()