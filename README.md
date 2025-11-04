# VIVEK: Vigilant Intelligent Verifier for Ethical Kids Content

A comprehensive **Deep Learning** project for content moderation using state-of-the-art neural networks for text, video, and audio analysis.

## 🎯 Overview

VIVEK is a full-fledged deep learning system designed to identify and filter inappropriate content for children. It uses advanced neural network architectures including:

- **Text Models**: LSTM, CNN, BERT-based transformers
- **Video Models**: CNN, ResNet, Transformer-based video classifiers
- **Audio Models**: CNN and Transformer-based audio classifiers
- **Multi-Modal Fusion**: Early fusion, Late fusion, and Attention-based fusion models

## 🚀 Features

- **Modular Architecture**: Clean, organized codebase with separate modules for models, data, training, and utilities
- **Multiple Model Types**: Support for text-only, video-only, audio-only, and multi-modal classification
- **Comprehensive Training**: Built-in support for training loops, checkpointing, early stopping, and learning rate scheduling
- **Configuration Management**: YAML-based configuration files for easy experiment management
- **Production Ready**: Inference scripts and model deployment utilities

## 📁 Project Structure

```
Vigilant-Intelligent-Verifier-for-Ethical-Kids-content/
├── src/
│   ├── models/              # Deep learning model definitions
│   │   ├── text_models.py    # Text classification models (LSTM, CNN, BERT)
│   │   ├── video_models.py   # Video classification models (CNN, ResNet, Transformer)
│   │   ├── audio_models.py   # Audio classification models (CNN, Transformer)
│   │   └── multimodal_models.py  # Multi-modal fusion models
│   ├── data/                 # Data loading and preprocessing
│   │   ├── datasets.py       # PyTorch dataset classes
│   │   ├── dataloaders.py    # Data loader utilities
│   │   └── transforms.py    # Data augmentation and transforms
│   ├── trainers/             # Training infrastructure
│   │   ├── trainer.py        # Main trainer class
│   │   ├── callbacks.py      # Training callbacks (early stopping, checkpointing)
│   │   └── metrics.py        # Metrics tracking
│   └── utils/                # Utility functions
│       ├── logging_utils.py  # Logging setup
│       ├── config_utils.py   # Configuration management
│       └── visualization.py  # Plotting utilities
├── configs/                  # Configuration files for different models
│   ├── train_text_lstm.yaml
│   ├── train_text_bert.yaml
│   ├── train_video_resnet.yaml
│   ├── train_audio_cnn.yaml
│   └── train_multimodal.yaml
├── data/                     # Data directory
│   ├── captions.csv          # Text captions
│   ├── labels.csv            # Ground truth labels
│   └── videos/               # Video files
├── checkpoints/              # Saved model checkpoints
├── logs/                     # Training logs
├── train.py                  # Main training script
├── inference.py              # Inference script
└── requirements.txt          # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- FFmpeg (for audio extraction from videos)

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install PyTorch** (if not already installed):
   - For CUDA 11.8: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
   - For CPU only: `pip install torch torchvision torchaudio`

5. **Install FFmpeg** (for audio processing):
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## 📊 Data Preparation

### Data Format

1. **Captions CSV** (`data/captions.csv`):
   - Required columns: `id`, `caption`
   - Example:
     ```csv
     id,caption
     vid001,"Kids playing in the park"
     vid002,"Educational cooking tutorial"
     ```

2. **Labels CSV** (`data/labels.csv`):
   - Required columns: `id`, `label`
   - Labels: `0` = Safe, `1` = Adult/Inappropriate
   - Example:
     ```csv
     id,label
     vid001,0
     vid002,0
     ```

3. **Video Files** (`data/videos/`):
   - Place video files in `data/videos/` directory
   - Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`
   - Video IDs should match those in `captions.csv` and `labels.csv`

## 🏋️ Training Models

### Training a Text Model

**LSTM Model**:
```bash
python train.py --config configs/train_text_lstm.yaml
```

**BERT Model**:
```bash
python train.py --config configs/train_text_bert.yaml
```

### Training a Video Model

```bash
python train.py --config configs/train_video_resnet.yaml
```

### Training an Audio Model

```bash
python train.py --config configs/train_audio_cnn.yaml
```

### Training a Multi-Modal Model

```bash
python train.py --config configs/train_multimodal.yaml
```

### Resuming Training

```bash
python train.py --config configs/train_text_lstm.yaml --resume checkpoints/text_lstm/checkpoint_epoch_10.pt
```

### Configuration

Edit the YAML files in `configs/` to customize:
- Model hyperparameters
- Training parameters (learning rate, batch size, epochs)
- Data paths
- Optimizer and scheduler settings
- Checkpoint and logging directories

## 🔍 Inference

### Text Prediction

```bash
python inference.py \
    --model checkpoints/text_bert/best_model.pt \
    --config configs/train_text_bert.yaml \
    --mode text \
    --text "Your text caption here"
```

### Video Prediction

```bash
python inference.py \
    --model checkpoints/video_resnet/best_model.pt \
    --config configs/train_video_resnet.yaml \
    --mode video \
    --video path/to/video.mp4
```

### Audio Prediction

```bash
python inference.py \
    --model checkpoints/audio_cnn/best_model.pt \
    --config configs/train_audio_cnn.yaml \
    --mode audio \
    --video path/to/video.mp4
```

## 📈 Model Architectures

### Text Models

1. **TextLSTM**: Bidirectional LSTM with embedding layer
2. **TextCNN**: Multi-filter CNN with different kernel sizes
3. **TextBERT**: BERT-based transformer model

### Video Models

1. **VideoCNN**: Frame-level CNN with temporal pooling
2. **VideoResNet**: ResNet backbone with LSTM for temporal modeling
3. **VideoTransformer**: Transformer-based video classifier

### Audio Models

1. **AudioCNN**: CNN on mel spectrograms
2. **AudioTransformer**: Transformer-based audio classifier

### Multi-Modal Models

1. **EarlyFusion**: Concatenate features before classification
2. **LateFusion**: Combine predictions from individual models
3. **MultiModalFusion**: Attention-based fusion mechanism

## 📊 Monitoring Training

Training progress is logged to:
- **Console**: Real-time metrics during training
- **Log files**: `logs/train_YYYYMMDD_HHMMSS.log`
- **Checkpoints**: Saved models in `checkpoints/`

Visualize training curves:
```python
from src.utils import plot_training_curves
import torch

checkpoint = torch.load('checkpoints/text_lstm/checkpoint_epoch_50.pt')
history = checkpoint['history']
plot_training_curves(history, save_path='training_curves.png')
```

## 🎯 Model Performance

The system tracks multiple metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for positive class
- **Recall**: Recall for positive class
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

## 🔧 Customization

### Adding New Models

1. Create model class in `src/models/`
2. Add model creation logic in `train.py`
3. Create configuration file in `configs/`

### Custom Data Loaders

1. Create dataset class inheriting from `torch.utils.data.Dataset`
2. Add loader function in `src/data/dataloaders.py`
3. Update `train.py` to use new loader

## 📝 Configuration Files

Configuration files use YAML format. Key sections:

- **model**: Model architecture and hyperparameters
- **data**: Data paths and preprocessing settings
- **training**: Training hyperparameters (epochs, learning rate, etc.)
- **checkpoint**: Checkpoint saving settings

Example:
```yaml
model:
  type: "TextBERT"
  model_name: "bert-base-uncased"
  num_classes: 2
  dropout: 0.3

data:
  captions_path: "data/captions.csv"
  labels_path: "data/labels.csv"
  batch_size: 16
  max_length: 128

training:
  num_epochs: 10
  learning_rate: 2e-5
  optimizer: "adamw"
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config file
- Use gradient accumulation (add to training config)
- Use mixed precision training

### Slow Training
- Reduce number of workers in data loader
- Use smaller model variants
- Enable mixed precision training

### Audio Extraction Fails
- Ensure FFmpeg is installed and in PATH
- Check video file format and codec
- Verify video files have audio tracks

## 📚 References

- PyTorch: https://pytorch.org/
- Transformers (HuggingFace): https://huggingface.co/transformers/
- Librosa: https://librosa.org/

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- HuggingFace for transformer models
- PyTorch team for the deep learning framework
- OpenCV and Librosa for multimedia processing

---

**Note**: This project is designed for content moderation research. Always ensure proper data handling and ethical considerations when working with sensitive content.
