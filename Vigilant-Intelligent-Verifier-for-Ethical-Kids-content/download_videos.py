import pandas as pd
import requests
import os
from tqdm import tqdm

# --- Configuration ---
# NOTE: You MUST create this CSV file with columns: 'id' and 'url'
URL_LIST_PATH = "data/video_urls.csv"
DOWNLOAD_DIR = "data/videos" 
# ---------------------

def download_video_dataset(url_list_path, download_dir):
    """
    Downloads videos from a list of URLs and saves them with their corresponding ID.
    The file extension is determined from the URL.
    """
    if not os.path.exists(url_list_path):
        print(f"❌ ERROR: URL list file not found at {url_list_path}. Please create it.")
        return

    try:
        urls_df = pd.read_csv(url_list_path)
    except Exception as e:
        print(f"❌ ERROR reading CSV: {e}")
        return

    if 'id' not in urls_df.columns or 'url' not in urls_df.columns:
        print("❌ ERROR: CSV must contain columns 'id' and 'url'.")
        return

    os.makedirs(download_dir, exist_ok=True)
    
    print(f"📦 Starting download of {len(urls_df)} videos to {download_dir}...")
    
    successful_downloads = 0
    
    for index, row in tqdm(urls_df.iterrows(), total=len(urls_df), desc="Downloading Videos"):
        video_id = row['id']
        video_url = row['url']
        
        # Simple attempt to get file extension from URL
        extension = os.path.splitext(video_url.split('?')[-2].split('/')[-1])[-1]
        
        # Default to .mp4 if extension is missing or too short
        if not extension or len(extension) > 5:
            extension = ".mp4"
            
        save_path = os.path.join(download_dir, f"{video_id}{extension}")

        if os.path.exists(save_path):
            # Skip if the file already exists
            successful_downloads += 1
            continue

        try:
            # Using stream=True for large files
            response = requests.get(video_url, stream=True, timeout=10)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

            # Write file in chunks to handle large videos efficiently
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            successful_downloads += 1
            
        except requests.exceptions.RequestException as e:
            print(f"\n⚠️  Failed to download {video_id} from {video_url}. Error: {e}")
        except Exception as e:
             print(f"\n❌ An unexpected error occurred for {video_id}: {e}")

    print("\n" + "="*50)
    print(f"✅ Download complete! {successful_downloads} of {len(urls_df)} files available.")
    print("="*50)

if __name__ == "__main__":
    download_video_dataset(URL_LIST_PATH, DOWNLOAD_DIR)