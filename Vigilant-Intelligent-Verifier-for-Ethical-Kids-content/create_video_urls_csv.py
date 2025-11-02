import pandas as pd
import os

OUTPUT_PATH = "data/video_urls.csv"

def generate_placeholder_urls(num_samples=300):
    """Generates placeholder IDs and URLs based on the expected number of samples."""
    
    data = []
    
    # Generate IDs from vid001 up to vid300
    for i in range(1, num_samples + 1):
        video_id = f"vid{i:03d}"
        
        # --- Placeholder URLs ---
        # NOTE: REPLACE THESE WITH YOUR ACTUAL DOWNLOAD LINKS (e.g., from Creative Commons/Research)
        if i % 2 == 1:
            # Assign a generic "safe" placeholder URL for odd IDs
            url = f"https://placeholder.com/download/safe/clip_{i}.mp4"
        else:
            # Assign a generic "unsafe" placeholder URL for even IDs
            url = f"https://placeholder.com/download/unsafe/clip_{i}.avi"
        # ------------------------
        
        data.append({'id': video_id, 'url': url})

    df = pd.DataFrame(data)
    
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    
    # Save the CSV
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"✅ Generated placeholder URL file with {len(df)} entries: {OUTPUT_PATH}")
    print("⚠️ WARNING: You MUST replace the placeholder URLs in this file with actual video download links.")

if __name__ == "__main__":
    generate_placeholder_urls()