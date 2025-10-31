import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

VIDEO_DIR = "data/videos"
OUTPUT = "features/video_features.csv"

def extract_video_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    brightness, motion = [], []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            motion.append(np.mean(diff))
        prev_frame = gray
    cap.release()

    return {
        "avg_brightness": np.mean(brightness),
        "motion_intensity": np.mean(motion) if motion else 0,
        "frame_count": frame_count
    }

def main():
    data = []
    for f in tqdm(os.listdir(VIDEO_DIR)):
        if f.endswith((".mp4", ".avi")):
            path = os.path.join(VIDEO_DIR, f)
            feats = extract_video_features(path)
            feats["id"] = os.path.splitext(f)[0]
            data.append(feats)

    df = pd.DataFrame(data)
    os.makedirs("features", exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print("✅ Video features saved to", OUTPUT)

if __name__ == "__main__":
    main()
