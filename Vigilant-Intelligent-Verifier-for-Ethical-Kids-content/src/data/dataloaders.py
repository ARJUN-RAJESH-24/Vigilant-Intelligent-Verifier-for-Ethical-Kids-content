"""
Data Loader Utilities
"""

from torch.utils.data import DataLoader
from .datasets import TextDataset, VideoDataset, AudioDataset, MultiModalDataset


def get_text_loader(captions_path, labels_path, tokenizer, batch_size=32,
                    max_length=128, shuffle=True, num_workers=4, split='train'):
    """
    Get text data loader
    """
    dataset = TextDataset(
        captions_path=captions_path,
        labels_path=labels_path,
        tokenizer=tokenizer,
        max_length=max_length,
        split=split
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_video_loader(video_dir, labels_path, batch_size=8, num_frames=16,
                     frame_size=(224, 224), shuffle=True, num_workers=2, split='train'):
    """
    Get video data loader
    """
    dataset = VideoDataset(
        video_dir=video_dir,
        labels_path=labels_path,
        num_frames=num_frames,
        frame_size=frame_size,
        split=split
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_audio_loader(video_dir, labels_path, batch_size=32, sample_rate=22050,
                     duration=30, n_mels=128, shuffle=True, num_workers=4, split='train'):
    """
    Get audio data loader
    """
    dataset = AudioDataset(
        video_dir=video_dir,
        labels_path=labels_path,
        sample_rate=sample_rate,
        duration=duration,
        n_mels=n_mels,
        split=split
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_multimodal_loader(captions_path, video_dir, labels_path, tokenizer,
                          batch_size=8, num_frames=16, frame_size=(224, 224),
                          audio_sample_rate=22050, audio_duration=30, audio_n_mels=128,
                          text_max_length=128, shuffle=True, num_workers=2, split='train'):
    """
    Get multi-modal data loader
    """
    dataset = MultiModalDataset(
        captions_path=captions_path,
        video_dir=video_dir,
        labels_path=labels_path,
        text_tokenizer=tokenizer,
        num_frames=num_frames,
        frame_size=frame_size,
        audio_sample_rate=audio_sample_rate,
        audio_duration=audio_duration,
        audio_n_mels=audio_n_mels,
        text_max_length=text_max_length,
        split=split
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader

