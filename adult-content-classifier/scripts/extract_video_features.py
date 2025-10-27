import cv2, numpy as np, os, pandas as pd
from tqdm import tqdm

def skin_ratio(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 40, 0], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return np.sum(mask > 0) / mask.size

def extract_video_features(video_path, n_frames=5):
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(length-1,1), n_frames).astype(int)
    features = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
        features.append([
            skin_ratio(frame),
            np.mean(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F))
        ])
    cap.release()
    if not features: return [0, 0, 0, 0]
    arr = np.array(features)
    return arr.mean(axis=0).tolist() + arr.std(axis=0).tolist()

def batch_extract(video_dir, out_csv):
    rows = []
    for f in tqdm(os.listdir(video_dir)):
        if not f.lower().endswith(('.mp4','.avi','.mov')): continue
        path = os.path.join(video_dir, f)
        rows.append([f] + extract_video_features(path))
    pd.DataFrame(rows, columns=['id','skin_mean','lap_mean','skin_std','lap_std']).to_csv(out_csv,index=False)

if __name__ == "__main__":
    os.makedirs('features', exist_ok=True)
    batch_extract('data/videos','features/video_features.csv')
