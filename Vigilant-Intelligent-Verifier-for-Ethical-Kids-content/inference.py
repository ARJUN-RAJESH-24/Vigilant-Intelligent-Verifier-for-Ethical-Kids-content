"""
Inference Script for Deep Learning Models
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import cv2
import librosa
from transformers import AutoTokenizer

# Import project modules
from src.models import (
    TextLSTM, TextBERT, TextCNN,
    VideoCNN, VideoResNet, VideoTransformer,
    AudioCNN, AudioTransformer,
    MultiModalFusion, EarlyFusion, LateFusion
)
from src.utils import load_config


def load_model(model_path, config_path, device='cuda'):
    """Load trained model"""
    config = load_config(config_path)
    model_config = config['model']
    model_type = model_config['type']
    
    # Create model
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
    
    elif model_type == "AudioCNN":
        model = AudioCNN(
            num_classes=model_config.get('num_classes', 2),
            dropout=model_config.get('dropout', 0.3)
        )
    
    else:
        raise ValueError(f"Model type {model_type} not yet supported for inference")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, config


def predict_text(model, text, config, device='cuda'):
    """Predict on text input"""
    model_type = config['model']['type']
    
    if model_type == "TextBERT":
        tokenizer = AutoTokenizer.from_pretrained(config['model'].get('model_name', 'bert-base-uncased'))
        encoded = tokenizer(
            text,
            max_length=config['data'].get('max_length', 128),
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1)
    
    else:
        # Simple tokenization for other models (placeholder)
        # In production, use proper tokenizer
        words = text.lower().split()
        tokens = [hash(word) % 10000 for word in words]
        tokens = tokens[:config['data'].get('max_length', 128)]
        tokens = tokens + [0] * (config['data'].get('max_length', 128) - len(tokens))
        
        input_ids = torch.tensor([tokens]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1)
    
    return pred.item(), probs[0].cpu().numpy()


def predict_video(model, video_path, config, device='cuda'):
    """Predict on video input"""
    num_frames = config['data'].get('num_frames', 16)
    frame_size = tuple(config['data'].get('frame_size', [224, 224]))
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return None, None
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(frame)
        else:
            frames.append(torch.zeros(3, frame_size[0], frame_size[1]))
    
    cap.release()
    
    frames = torch.stack(frames).unsqueeze(0).to(device)  # (1, num_frames, C, H, W)
    
    with torch.no_grad():
        outputs = model(frames)
        probs = F.softmax(outputs, dim=1)
        pred = torch.argmax(outputs, dim=1)
    
    return pred.item(), probs[0].cpu().numpy()


def predict_audio(model, video_path, config, device='cuda'):
    """Predict on audio input"""
    sample_rate = config['data'].get('sample_rate', 22050)
    duration = config['data'].get('duration', 30)
    n_mels = config['data'].get('n_mels', 128)
    
    try:
        # Extract audio
        audio, sr = librosa.load(video_path, sr=sample_rate, duration=duration, mono=True)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            hop_length=512
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Convert to tensor
        mel_spec_tensor = torch.from_numpy(mel_spec_db).float().unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(mel_spec_tensor)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1)
        
        return pred.item(), probs[0].cpu().numpy()
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Inference with Deep Learning Models')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--text', type=str, default=None, help='Text input for prediction')
    parser.add_argument('--video', type=str, default=None, help='Video file path for prediction')
    parser.add_argument('--mode', type=str, choices=['text', 'video', 'audio'], required=True,
                       help='Prediction mode')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Setup device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, config = load_model(args.model, args.config, device=device)
    print("✅ Model loaded successfully!")
    
    # Make prediction
    if args.mode == 'text':
        if not args.text:
            print("Error: --text argument required for text prediction")
            return
        
        print(f"\nPredicting on text: {args.text}")
        pred, probs = predict_text(model, args.text, config, device=device)
        
        class_names = ['Safe', 'Adult']
        print(f"\nPrediction: {class_names[pred]}")
        print(f"Confidence: {probs[pred]:.4f}")
        print(f"\nProbabilities:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {probs[i]:.4f}")
    
    elif args.mode == 'video':
        if not args.video:
            print("Error: --video argument required for video prediction")
            return
        
        print(f"\nPredicting on video: {args.video}")
        pred, probs = predict_video(model, args.video, config, device=device)
        
        if pred is not None:
            class_names = ['Safe', 'Adult']
            print(f"\nPrediction: {class_names[pred]}")
            print(f"Confidence: {probs[pred]:.4f}")
            print(f"\nProbabilities:")
            for i, class_name in enumerate(class_names):
                print(f"  {class_name}: {probs[i]:.4f}")
        else:
            print("Error: Failed to process video")
    
    elif args.mode == 'audio':
        if not args.video:
            print("Error: --video argument required for audio prediction")
            return
        
        print(f"\nPredicting on audio from video: {args.video}")
        pred, probs = predict_audio(model, args.video, config, device=device)
        
        if pred is not None:
            class_names = ['Safe', 'Adult']
            print(f"\nPrediction: {class_names[pred]}")
            print(f"Confidence: {probs[pred]:.4f}")
            print(f"\nProbabilities:")
            for i, class_name in enumerate(class_names):
                print(f"  {class_name}: {probs[i]:.4f}")
        else:
            print("Error: Failed to process audio")


if __name__ == '__main__':
    main()

