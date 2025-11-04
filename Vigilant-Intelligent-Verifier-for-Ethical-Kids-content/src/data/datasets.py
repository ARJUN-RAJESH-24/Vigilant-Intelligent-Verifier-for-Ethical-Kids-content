"""
PyTorch Datasets for Content Moderation
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import librosa
from PIL import Image
import os
from typing import Optional, Tuple, List


class TextDataset(Dataset):
    """
    Dataset for text classification
    """
    
    def __init__(self, captions_path, labels_path, tokenizer, max_length=128, 
                 transform=None, split='train'):
        self.captions_df = pd.read_csv(captions_path)
        self.labels_df = pd.read_csv(labels_path)
        
        # Merge captions and labels
        self.df = self.captions_df.merge(self.labels_df, on='id', how='inner')
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform
        self.split = split
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get caption
        caption = str(row['caption']) if pd.notna(row['caption']) else ""
        
        # Tokenize
        if hasattr(self.tokenizer, 'encode'):
            # BERT tokenizer
            encoded = self.tokenizer.encode(
                caption,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded.squeeze(0)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        else:
            # Simple tokenizer (vocab-based)
            tokens = self.tokenizer(caption)
            input_ids = torch.tensor(tokens[:self.max_length] + [0] * (self.max_length - len(tokens)))
            attention_mask = (input_ids != 0).long()
        
        # Get label
        label = int(row['label'])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'caption': caption
        }


class VideoDataset(Dataset):
    """
    Dataset for video classification
    """
    
    def __init__(self, video_dir, labels_path, num_frames=16, frame_size=(224, 224),
                 transform=None, split='train'):
        self.video_dir = video_dir
        self.labels_df = pd.read_csv(labels_path)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.split = split
        
        # Get list of video files
        self.video_files = [f for f in os.listdir(video_dir) 
                           if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Create mapping from video ID to label
        self.label_map = dict(zip(self.labels_df['id'], self.labels_df['label']))
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        
        # Get label
        label = self.label_map.get(video_id, 0)
        
        # Extract frames
        frames = self._extract_frames(video_path)
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'frames': frames,
            'label': torch.tensor(label, dtype=torch.long),
            'video_id': video_id
        }
    
    def _extract_frames(self, video_path):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            # Return black frames if video is empty
            return torch.zeros(self.num_frames, 3, self.frame_size[0], self.frame_size[1])
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame
                frame = cv2.resize(frame, self.frame_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                # Convert to tensor and transpose to (C, H, W)
                frame = torch.from_numpy(frame).permute(2, 0, 1)
                frames.append(frame)
            else:
                # Use black frame if read fails
                frames.append(torch.zeros(3, self.frame_size[0], self.frame_size[1]))
        
        cap.release()
        
        # Stack frames
        frames = torch.stack(frames)  # (num_frames, C, H, W)
        
        return frames


class AudioDataset(Dataset):
    """
    Dataset for audio classification
    """
    
    def __init__(self, video_dir, labels_path, sample_rate=22050, duration=30,
                 n_mels=128, transform=None, split='train'):
        self.video_dir = video_dir
        self.labels_df = pd.read_csv(labels_path)
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.transform = transform
        self.split = split
        
        # Get list of video files
        self.video_files = [f for f in os.listdir(video_dir) 
                           if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Create mapping from video ID to label
        self.label_map = dict(zip(self.labels_df['id'], self.labels_df['label']))
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)
        video_id = os.path.splitext(video_file)[0]
        
        # Get label
        label = self.label_map.get(video_id, 0)
        
        # Extract audio and convert to spectrogram
        spectrogram = self._extract_spectrogram(video_path)
        
        # Apply transforms
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return {
            'spectrogram': spectrogram,
            'label': torch.tensor(label, dtype=torch.long),
            'video_id': video_id
        }
    
    def _extract_spectrogram(self, video_path):
        """Extract audio and convert to mel spectrogram"""
        try:
            # Extract audio using librosa
            audio, sr = librosa.load(video_path, sr=self.sample_rate, duration=self.duration, mono=True)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                hop_length=512
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1]
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            # Convert to tensor and add channel dimension
            mel_spec_tensor = torch.from_numpy(mel_spec_db).float().unsqueeze(0)  # (1, n_mels, time)
            
            return mel_spec_tensor
            
        except Exception as e:
            # Return zeros if extraction fails
            print(f"Warning: Failed to extract audio from {video_path}: {e}")
            return torch.zeros(1, self.n_mels, 1000)  # Default size


class MultiModalDataset(Dataset):
    """
    Multi-modal dataset combining text, video, and audio
    """
    
    def __init__(self, captions_path, video_dir, labels_path, 
                 text_tokenizer, num_frames=16, frame_size=(224, 224),
                 audio_sample_rate=22050, audio_duration=30, audio_n_mels=128,
                 text_max_length=128, transform=None, split='train'):
        
        # Initialize individual datasets
        self.text_dataset = TextDataset(
            captions_path, labels_path, text_tokenizer, 
            max_length=text_max_length, split=split
        )
        
        self.video_dataset = VideoDataset(
            video_dir, labels_path, num_frames=num_frames,
            frame_size=frame_size, split=split
        )
        
        self.audio_dataset = AudioDataset(
            video_dir, labels_path, sample_rate=audio_sample_rate,
            duration=audio_duration, n_mels=audio_n_mels, split=split
        )
        
        # Ensure all datasets have same length
        min_len = min(len(self.text_dataset), len(self.video_dataset), len(self.audio_dataset))
        self.length = min_len
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        text_item = self.text_dataset[idx % len(self.text_dataset)]
        video_item = self.video_dataset[idx % len(self.video_dataset)]
        audio_item = self.audio_dataset[idx % len(self.audio_dataset)]
        
        return {
            'text': text_item,
            'video': video_item,
            'audio': audio_item,
            'label': text_item['label']  # All should have same label
        }

