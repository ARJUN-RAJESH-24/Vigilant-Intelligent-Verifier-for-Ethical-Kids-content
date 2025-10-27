import librosa, os, pandas as pd, numpy as np
from tqdm import tqdm

def extract_audio_features(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    energy = np.mean(librosa.feature.rms(y=y))
    return [np.mean(mfcc), np.std(mfcc), np.mean(zcr), energy]

def batch_extract(audio_dir, out_csv):
    rows = []
    for f in tqdm(os.listdir(audio_dir)):
        if not f.lower().endswith(('.wav','.mp3')): continue
        path = os.path.join(audio_dir, f)
        rows.append([f] + extract_audio_features(path))
    pd.DataFrame(rows, columns=['id','mfcc_mean','mfcc_std','zcr_mean','energy']).to_csv(out_csv,index=False)

if __name__ == "__main__":
    os.makedirs('features', exist_ok=True)
    batch_extract('data/audio','features/audio_features.csv')
