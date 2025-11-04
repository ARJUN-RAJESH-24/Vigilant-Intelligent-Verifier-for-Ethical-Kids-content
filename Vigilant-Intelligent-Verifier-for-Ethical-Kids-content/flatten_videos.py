import os
import shutil
from tqdm import tqdm

VIDEO_DIR = "data/videos"
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.webm', '.mkv')
MIN_FILE_SIZE_BYTES = 1000 # Skip very small, potentially corrupt files

def flatten_directory_structure():
    """
    Recursively searches subdirectories of VIDEO_DIR and moves all video files 
    to the root VIDEO_DIR, effectively flattening the structure.
    """
    
    if not os.path.isdir(VIDEO_DIR):
        print(f"❌ ERROR: Directory not found: {VIDEO_DIR}")
        return

    print(f"🎬 Starting recursive search in {VIDEO_DIR}...")
    
    files_moved = 0
    subfolders_to_remove = []
    
    # Use os.walk to traverse the directory tree
    for root, dirs, files in os.walk(VIDEO_DIR, topdown=False):
        # Exclude the root directory itself from processing/deletion
        if root == VIDEO_DIR:
            continue
            
        for name in files:
            file_path = os.path.join(root, name)
            
            # Check if the file is a video and is large enough
            if name.lower().endswith(VIDEO_EXTENSIONS) and os.path.getsize(file_path) > MIN_FILE_SIZE_BYTES:
                
                # New path is just the filename in the root directory
                new_file_path = os.path.join(VIDEO_DIR, name)
                
                # Check for collision, though unlikely with hex names
                if os.path.exists(new_file_path):
                    print(f"⚠️ Skipping {name}: File already exists in root.")
                    continue
                
                try:
                    shutil.move(file_path, new_file_path)
                    files_moved += 1
                except Exception as e:
                    print(f"❌ Error moving file {name}: {e}")
            
        # Collect subfolders that might be empty after file moves
        # This is safe because os.walk is topdown=False (bottom-up traversal)
        subfolders_to_remove.append(root)

    print("\n" + "="*50)
    print(f"✅ Finished moving files. Total files moved: {files_moved}")
    
    # Attempt to remove empty subfolders
    for folder in tqdm(subfolders_to_remove, desc="Removing empty subfolders"):
        try:
            os.rmdir(folder)
        except OSError:
            # OSError usually means the folder is not empty
            pass

    print("✅ Directory structure flattened.")
    
    if files_moved > 0:
        print("\n➡️ NEXT STEP: Rename the first 300 files in data/videos/ to match your IDs (vid001.mp4 through vid300.mp4).")


if __name__ == "__main__":
    flatten_directory_structure()