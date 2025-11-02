import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

AUDIO_DIR = "data/videos"
OUTPUT = "features/audio_features.csv"

def extract_audio_features(file_path):
    """
    Extract comprehensive audio features from video/audio files.
    
    Features extracted:
    - RMS (Root Mean Square) energy
    - Zero Crossing Rate
    - Spectral Centroid
    - Spectral Bandwidth
    - Spectral Rolloff
    - MFCC (Mel-Frequency Cepstral Coefficients) statistics
    - Chroma features
    - Tempo
    """
    try:
        # Load audio with librosa (mono, first 30 seconds for efficiency)
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
        
        # Handle empty audio
        if len(y) == 0:
            print(f"⚠️ Warning: Empty audio in {file_path}")
            return None
        
        # Basic features
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # MFCC features (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Compile features
        features = {
            "rms_mean": np.mean(rms),
            "rms_std": np.std(rms),
            "zcr_mean": np.mean(zcr),
            "zcr_std": np.std(zcr),
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_centroid_std": np.std(spectral_centroid),
            "spectral_bandwidth_mean": np.mean(spectral_bandwidth),
            "spectral_bandwidth_std": np.std(spectral_bandwidth),
            "spectral_rolloff_mean": np.mean(spectral_rolloff),
            "spectral_rolloff_std": np.std(spectral_rolloff),
            "tempo": float(tempo),
        }
        
        # Add MFCC statistics
        for i in range(13):
            features[f"mfcc_{i}_mean"] = np.mean(mfcc[i])
            features[f"mfcc_{i}_std"] = np.std(mfcc[i])
        
        # Add Chroma statistics
        features["chroma_mean"] = np.mean(chroma)
        features["chroma_std"] = np.std(chroma)
        
        return features
        
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

def main():
    """Main function to extract audio features from all files."""
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Error: Directory {AUDIO_DIR} does not exist!")
        return
    
    data = []
    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith((".mp4", ".mp3", ".wav", ".avi"))]
    
    if not files:
        print(f"⚠️ Warning: No audio/video files found in {AUDIO_DIR}")
        return
    
    print(f"📊 Processing {len(files)} audio files...")
    
    for file in tqdm(files, desc="Extracting audio features"):
        path = os.path.join(AUDIO_DIR, file)
        feats = extract_audio_features(path)
        
        if feats:
            feats["id"] = os.path.splitext(file)[0]
            data.append(feats)
    
    if not data:
        print("❌ No features extracted. Check your audio files.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns to have 'id' first
    cols = ['id'] + [col for col in df.columns if col != 'id']
    df = df[cols]
    
    # Create output directory
    os.makedirs("features", exist_ok=True)
    
    # Save to CSV
    df.to_csv(OUTPUT, index=False)
    print(f"✅ Audio features saved to {OUTPUT}")
    print(f"📈 Extracted {len(df)} samples with {len(df.columns)-1} features each")

if __name__ == "__main__":
    main()