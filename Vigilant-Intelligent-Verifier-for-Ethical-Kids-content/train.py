"""
Main Training Script for Deep Learning Models
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path

# Import project modules
from src.models import (
    TextLSTM, TextBERT, TextCNN,
    VideoCNN, VideoResNet, VideoTransformer,
    AudioCNN, AudioTransformer,
    MultiModalFusion, EarlyFusion, LateFusion
)
from src.data import get_text_loader, get_video_loader, get_audio_loader, get_multimodal_loader
from src.trainers import Trainer
from src.utils import load_config, setup_logger, plot_training_curves
from transformers import AutoTokenizer


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(config, device):
    """Create model based on config"""
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == "TextLSTM":
        model = TextLSTM(
            vocab_size=model_config.get('vocab_size', 10000),
            embedding_dim=model_config.get('embedding_dim', 128),
            hidden_dim=model_config.get('hidden_dim', 256),
            num_layers=model_config.get('num_layers', 2),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    
    elif model_type == "TextBERT":
        model = TextBERT(
            model_name=model_config.get('model_name', 'bert-base-uncased'),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    
    elif model_type == "TextCNN":
        model = TextCNN(
            vocab_size=model_config.get('vocab_size', 10000),
            embedding_dim=model_config.get('embedding_dim', 128),
            num_filters=model_config.get('num_filters', 100),
            filter_sizes=model_config.get('filter_sizes', [3, 4, 5]),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    
    elif model_type == "VideoCNN":
        model = VideoCNN(
            num_frames=config['data'].get('num_frames', 16),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    
    elif model_type == "VideoResNet":
        model = VideoResNet(
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3),
            model_name=model_config.get('model_name', 'resnet18')
        )
    
    elif model_type == "VideoTransformer":
        model = VideoTransformer(
            num_frames=config['data'].get('num_frames', 16),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    
    elif model_type == "AudioCNN":
        model = AudioCNN(
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    
    elif model_type == "AudioTransformer":
        model = AudioTransformer(
            input_dim=config['data'].get('n_mels', 128),
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    
    elif model_type == "MultiModalFusion":
        fusion_type = model_config.get('fusion_type', 'attention')
        
        if fusion_type == 'early':
            model = EarlyFusion(
                text_dim=model_config.get('text_dim', 768),
                video_dim=model_config.get('video_dim', 512),
                audio_dim=model_config.get('audio_dim', 256),
                num_classes=model_config.get('num_classes', 2),
                dropout=model_config.get('dropout', 0.3)
            )
        elif fusion_type == 'late':
            model = LateFusion(
                text_dim=model_config.get('text_dim', 768),
                video_dim=model_config.get('video_dim', 512),
                audio_dim=model_config.get('audio_dim', 256),
                num_classes=model_config.get('num_classes', 2),
                dropout=model_config.get('dropout', 0.3)
            )
        else:  # attention
            model = MultiModalFusion(
                text_dim=model_config.get('text_dim', 768),
                video_dim=model_config.get('video_dim', 512),
                audio_dim=model_config.get('audio_dim', 256),
                num_classes=model_config.get('num_classes', 2),
                dropout=model_config.get('dropout', 0.3)
            )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def get_tokenizer(model_type, model_name=None):
    """Get appropriate tokenizer"""
    if model_type in ["TextBERT"]:
        if model_name:
            return AutoTokenizer.from_pretrained(model_name)
        return AutoTokenizer.from_pretrained('bert-base-uncased')
    else:
        # Simple vocabulary-based tokenizer (placeholder)
        # In production, use a proper tokenizer
        class SimpleTokenizer:
            def __call__(self, text):
                # Simple word-based tokenization
                words = text.lower().split()
                # Map words to indices (simplified)
                return [hash(word) % 10000 for word in words]
        return SimpleTokenizer()


def main():
    parser = argparse.ArgumentParser(description='Train Deep Learning Models for Content Moderation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logger('train', log_dir='logs')
    logger.info(f"Starting training with config: {args.config}")
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Setup device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating {config['model']['type']} model...")
    model = create_model(config, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Get tokenizer (for text models)
    data_config = config['data']
    model_type = config['model']['type']
    
    if model_type in ["TextLSTM", "TextBERT", "TextCNN", "MultiModalFusion"]:
        tokenizer = get_tokenizer(model_type, config['model'].get('model_name'))
    else:
        tokenizer = None
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    if model_type in ["TextLSTM", "TextBERT", "TextCNN"]:
        train_loader = get_text_loader(
            captions_path=data_config['captions_path'],
            labels_path=data_config['labels_path'],
            tokenizer=tokenizer,
            batch_size=data_config['batch_size'],
            max_length=data_config['max_length'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            split='train'
        )
        
        val_loader = get_text_loader(
            captions_path=data_config['captions_path'],
            labels_path=data_config['labels_path'],
            tokenizer=tokenizer,
            batch_size=data_config['batch_size'],
            max_length=data_config['max_length'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            split='val'
        )
    
    elif model_type in ["VideoCNN", "VideoResNet", "VideoTransformer"]:
        train_loader = get_video_loader(
            video_dir=data_config['video_dir'],
            labels_path=data_config['labels_path'],
            batch_size=data_config['batch_size'],
            num_frames=data_config.get('num_frames', 16),
            frame_size=tuple(data_config.get('frame_size', [224, 224])),
            shuffle=True,
            num_workers=data_config.get('num_workers', 2),
            split='train'
        )
        
        val_loader = get_video_loader(
            video_dir=data_config['video_dir'],
            labels_path=data_config['labels_path'],
            batch_size=data_config['batch_size'],
            num_frames=data_config.get('num_frames', 16),
            frame_size=tuple(data_config.get('frame_size', [224, 224])),
            shuffle=False,
            num_workers=data_config.get('num_workers', 2),
            split='val'
        )
    
    elif model_type in ["AudioCNN", "AudioTransformer"]:
        train_loader = get_audio_loader(
            video_dir=data_config['video_dir'],
            labels_path=data_config['labels_path'],
            batch_size=data_config['batch_size'],
            sample_rate=data_config.get('sample_rate', 22050),
            duration=data_config.get('duration', 30),
            n_mels=data_config.get('n_mels', 128),
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            split='train'
        )
        
        val_loader = get_audio_loader(
            video_dir=data_config['video_dir'],
            labels_path=data_config['labels_path'],
            batch_size=data_config['batch_size'],
            sample_rate=data_config.get('sample_rate', 22050),
            duration=data_config.get('duration', 30),
            n_mels=data_config.get('n_mels', 128),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            split='val'
        )
    
    elif model_type == "MultiModalFusion":
        train_loader = get_multimodal_loader(
            captions_path=data_config['captions_path'],
            video_dir=data_config['video_dir'],
            labels_path=data_config['labels_path'],
            tokenizer=tokenizer,
            batch_size=data_config['batch_size'],
            num_frames=data_config.get('num_frames', 16),
            frame_size=tuple(data_config.get('frame_size', [224, 224])),
            audio_sample_rate=data_config.get('audio_sample_rate', 22050),
            audio_duration=data_config.get('audio_duration', 30),
            audio_n_mels=data_config.get('audio_n_mels', 128),
            text_max_length=data_config.get('text_max_length', 128),
            shuffle=True,
            num_workers=data_config.get('num_workers', 2),
            split='train'
        )
        
        val_loader = get_multimodal_loader(
            captions_path=data_config['captions_path'],
            video_dir=data_config['video_dir'],
            labels_path=data_config['labels_path'],
            tokenizer=tokenizer,
            batch_size=data_config['batch_size'],
            num_frames=data_config.get('num_frames', 16),
            frame_size=tuple(data_config.get('frame_size', [224, 224])),
            audio_sample_rate=data_config.get('audio_sample_rate', 22050),
            audio_duration=data_config.get('audio_duration', 30),
            audio_n_mels=data_config.get('audio_n_mels', 128),
            text_max_length=data_config.get('text_max_length', 128),
            shuffle=False,
            num_workers=data_config.get('num_workers', 2),
            split='val'
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device=device)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(config['training']['num_epochs'])
    
    # Plot training curves
    plot_training_curves(trainer.history, save_path=f"{config['checkpoint']['checkpoint_dir']}/training_curves.png")
    
    # Save final model
    final_model_path = f"{config['checkpoint']['checkpoint_dir']}/final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': trainer.history
    }, final_model_path)
    
    logger.info(f"✅ Training completed! Final model saved to: {final_model_path}")


if __name__ == '__main__':
    main()

