"""
Data Transformations and Augmentations
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np


class TextTransform:
    """
    Text transformation utilities
    """
    
    @staticmethod
    def normalize_text(text):
        """Normalize text"""
        return text.lower().strip()


class VideoTransform:
    """
    Video transformation and augmentation
    """
    
    def __init__(self, is_training=True):
        self.is_training = is_training
        
        if is_training:
            # Training augmentations
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/test: only normalization
            self.transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, frames):
        """
        Apply transforms to video frames
        frames: (num_frames, C, H, W)
        """
        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            # Convert to PIL Image for transforms
            frame_pil = F.to_pil_image(frame)
            transformed_frame = self.transform(frame_pil)
            transformed_frames.append(transformed_frame)
        
        return torch.stack(transformed_frames)


class AudioTransform:
    """
    Audio transformation and augmentation
    """
    
    def __init__(self, is_training=True):
        self.is_training = is_training
    
    def __call__(self, spectrogram):
        """
        Apply transforms to audio spectrogram
        spectrogram: (1, n_mels, time)
        """
        if self.is_training:
            # Add noise
            noise = torch.randn_like(spectrogram) * 0.01
            spectrogram = spectrogram + noise
            
            # Time masking
            if np.random.random() < 0.5:
                time_mask = np.random.randint(0, spectrogram.size(2) // 4)
                time_start = np.random.randint(0, spectrogram.size(2) - time_mask)
                spectrogram[:, :, time_start:time_start + time_mask] = 0
            
            # Frequency masking
            if np.random.random() < 0.5:
                freq_mask = np.random.randint(0, spectrogram.size(1) // 4)
                freq_start = np.random.randint(0, spectrogram.size(1) - freq_mask)
                spectrogram[:, freq_start:freq_start + freq_mask, :] = 0
        
        return spectrogram

