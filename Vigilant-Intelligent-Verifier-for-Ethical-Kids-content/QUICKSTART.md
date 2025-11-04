# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup Environment

**Linux/macOS:**
```bash
chmod +x setup_project.sh
./setup_project.sh
source venv/bin/activate
```

**Windows:**
```cmd
setup_project.bat
venv\Scripts\activate
```

### 2. Prepare Your Data

Ensure you have:
- `data/captions.csv` with columns: `id`, `caption`
- `data/labels.csv` with columns: `id`, `label` (0=Safe, 1=Adult)
- Video files in `data/videos/` (optional, for video/audio models)

### 3. Train Your First Model

**Text LSTM Model:**
```bash
python train.py --config configs/train_text_lstm.yaml
```

**Text BERT Model (Recommended):**
```bash
python train.py --config configs/train_text_bert.yaml
```

### 4. Make Predictions

**Text Prediction:**
```bash
python inference.py \
    --model checkpoints/text_lstm/best_model.pt \
    --config configs/train_text_lstm.yaml \
    --mode text \
    --text "Your text here"
```

## 📝 Example Workflow

### Training a Text Model

1. **Prepare data:**
   ```python
   import pandas as pd
   
   captions = pd.DataFrame({
       'id': ['vid001', 'vid002'],
       'caption': ['Kids playing', 'Family dinner']
   })
   captions.to_csv('data/captions.csv', index=False)
   
   labels = pd.DataFrame({
       'id': ['vid001', 'vid002'],
       'label': [0, 0]  # Both safe
   })
   labels.to_csv('data/labels.csv', index=False)
   ```

2. **Train model:**
   ```bash
   python train.py --config configs/train_text_bert.yaml
   ```

3. **Check results:**
   - Check `checkpoints/text_bert/` for saved models
   - Check `logs/` for training logs
   - View training curves in saved plots

### Making Predictions

```bash
# Text prediction
python inference.py \
    --model checkpoints/text_bert/best_model.pt \
    --config configs/train_text_bert.yaml \
    --mode text \
    --text "Kids playing soccer in the park"

# Output:
# Prediction: Safe
# Confidence: 0.9542
```

## 🔧 Customization

### Modify Training Parameters

Edit `configs/train_text_bert.yaml`:

```yaml
training:
  num_epochs: 20          # Change number of epochs
  learning_rate: 1e-5     # Change learning rate
  batch_size: 32          # Change batch size
```

### Use Different Model

Change model type in config:
```yaml
model:
  type: "TextLSTM"  # or "TextCNN", "TextBERT"
```

## 📊 Monitor Training

Training outputs:
- **Console**: Real-time metrics
- **Logs**: `logs/train_*.log`
- **Checkpoints**: `checkpoints/*/checkpoint_*.pt`

## 🐛 Common Issues

### CUDA Out of Memory
Reduce batch size in config:
```yaml
data:
  batch_size: 8  # Reduce from 16 or 32
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### FFmpeg Not Found
Install FFmpeg:
- Ubuntu: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from ffmpeg.org

## 📚 Next Steps

1. **Experiment with different models**: Try TextCNN, VideoResNet, etc.
2. **Tune hyperparameters**: Adjust learning rate, batch size, etc.
3. **Try multi-modal fusion**: Combine text, video, and audio
4. **Read the full README.md** for advanced usage

---

**Need Help?** Check the main README.md for detailed documentation.

