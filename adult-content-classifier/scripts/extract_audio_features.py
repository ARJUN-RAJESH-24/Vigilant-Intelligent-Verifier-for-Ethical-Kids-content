import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

AUDIO_DIR = "data/videos"
OUTPUT = "features/audio_features.csv"

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True, duration=10)
        features = {
            "rms": np.mean(librosa.feature.rms(y=y)),
            "zcr": np.mean(librosa.feature.zero_crossing_rate(y)),
            "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "spectral_bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        }
        return features
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

def main():
    data = []
    for file in tqdm(os.listdir(AUDIO_DIR)):
        if file.endswith((".mp4", ".mp3")):
            path = os.path.join(AUDIO_DIR, file)
            feats = extract_audio_features(path)
            if feats:
                feats["id"] = os.path.splitext(file)[0]
                data.append(feats)
    df = pd.DataFrame(data)
    os.makedirs("features", exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print("✅ Audio features saved to", OUTPUT)

if __name__ == "__main__":
    main()
