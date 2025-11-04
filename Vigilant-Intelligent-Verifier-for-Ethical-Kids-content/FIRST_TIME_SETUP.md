# First Time Setup Guide - VIVEK Deep Learning Project

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] CUDA-capable GPU (recommended) or CPU
- [ ] At least 10GB free disk space
- [ ] Internet connection (for downloading models and dependencies)
- [ ] FFmpeg installed (for audio/video processing)

### Check Python Version
```cmd
python --version
```
Should show Python 3.8 or higher.

### Check if GPU is Available (Optional)
```cmd
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🚀 Step-by-Step Setup

### Step 1: Navigate to Project Directory

```cmd
cd "D:\Notes and Projects\Project\AI\VIVEK\Vigilant-Intelligent-Verifier-for-Ethical-Kids-content"
```

### Step 2: Create Virtual Environment

```cmd
python -m venv venv
```

This creates a virtual environment in the `venv` folder.

### Step 3: Activate Virtual Environment

```cmd
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt.

### Step 4: Upgrade pip

```cmd
python -m pip install --upgrade pip
```

### Step 5: Install PyTorch (Choose one based on your system)

**For CUDA (GPU support):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```cmd
pip install torch torchvision torchaudio
```

### Step 6: Install Project Dependencies

```cmd
pip install -r requirements.txt
```

This will install all required packages. It may take 5-10 minutes.

### Step 7: Install FFmpeg (for video/audio processing)

**Option A: Using Chocolatey (if installed):**
```cmd
choco install ffmpeg
```

**Option B: Manual Installation:**
1. Download from: https://ffmpeg.org/download.html
2. Extract and add to system PATH
3. Verify: `ffmpeg -version`

**Option C: Skip for now** (if only using text models)

### Step 8: Create Required Directories

```cmd
mkdir data\videos
mkdir features
mkdir checkpoints
mkdir logs
mkdir results
```

Or use the setup script:
```cmd
setup_project.bat
```

### Step 9: Prepare Your Data

#### Option A: Use Existing Data
If you already have:
- `data/captions.csv`
- `data/labels.csv`
- Video files in `data/videos/`

Skip to Step 10.

#### Option B: Create Sample Data for Testing

Create `data/captions.csv`:
```python
import pandas as pd

captions = pd.DataFrame({
    'id': ['vid001', 'vid002', 'vid003', 'vid004', 'vid005'],
    'caption': [
        'Kids playing soccer in the park',
        'Family cooking dinner together',
        'Educational science experiment',
        'Hot dance performance at nightclub',
        'Romantic couple intimate scene'
    ]
})

captions.to_csv('data/captions.csv', index=False)
print("✅ Created captions.csv")
```

Create `data/labels.csv`:
```python
labels = pd.DataFrame({
    'id': ['vid001', 'vid002', 'vid003', 'vid004', 'vid005'],
    'label': [0, 0, 0, 1, 1]  # 0=Safe, 1=Adult
})

labels.to_csv('data/labels.csv', index=False)
print("✅ Created labels.csv")
```

Or run this Python script:
```cmd
python -c "import pandas as pd; captions = pd.DataFrame({'id': ['vid001', 'vid002', 'vid003', 'vid004', 'vid005'], 'caption': ['Kids playing soccer in the park', 'Family cooking dinner together', 'Educational science experiment', 'Hot dance performance at nightclub', 'Romantic couple intimate scene']}); captions.to_csv('data/captions.csv', index=False); labels = pd.DataFrame({'id': ['vid001', 'vid002', 'vid003', 'vid004', 'vid005'], 'label': [0, 0, 0, 1, 1]}); labels.to_csv('data/labels.csv', index=False); print('✅ Sample data created')"
```

## 🎯 Step 10: First Training Run

### Recommended: Start with Text BERT Model (Easiest)

```cmd
python train.py --config configs/train_text_bert.yaml
```

**What to expect:**
- Model will download BERT weights (first time only)
- Training will start showing progress bars
- Metrics will be printed after each epoch
- Checkpoints saved in `checkpoints/text_bert/`

### Alternative: Text LSTM (Faster, less memory)

```cmd
python train.py --config configs/train_text_lstm.yaml
```

### If You Get Errors:

**CUDA Out of Memory:**
- Edit `configs/train_text_bert.yaml`
- Change `batch_size: 16` to `batch_size: 8` or `batch_size: 4`

**Missing Dependencies:**
```cmd
pip install transformers tokenizers
```

**FFmpeg Not Found (for video/audio models):**
- Install FFmpeg or skip video/audio models for now

## 🔍 Step 11: Make Your First Prediction

After training completes (or if you have a pre-trained model):

```cmd
python inference.py --model checkpoints/text_bert/best_model.pt --config configs/train_text_bert.yaml --mode text --text "Kids playing soccer in the park"
```

**Expected output:**
```
Using device: cuda (or cpu)
Loading model from checkpoints/text_bert/best_model.pt...
✅ Model loaded successfully!

Predicting on text: Kids playing soccer in the park

Prediction: Safe
Confidence: 0.9542

Probabilities:
  Safe: 0.9542
  Adult: 0.0458
```

## 📊 Step 12: Check Results

### Training Results:
- **Checkpoints**: `checkpoints/text_bert/`
- **Logs**: `logs/train_*.log`
- **Training Curves**: Check checkpoint directory for plots

### View Training Logs:
```cmd
type logs\train_*.log
```

Or open the most recent log file in a text editor.

## 🎓 Understanding the Output

### During Training:
```
Epoch 1/10
Training: 100%|████████| 10/10 [00:30<00:00, loss=0.5234]
Validation: 100%|████████| 3/3 [00:05<00:00]

Train Loss: 0.5234 | Train Acc: 0.8500
Val Loss: 0.4123 | Val Acc: 0.9000 | Val F1: 0.8750 | Val AUC: 0.9200
✅ New best model! Val Acc: 0.9000
```

### Model Checkpoints:
- `checkpoint_epoch_X.pt` - Saved every N epochs
- `best_model.pt` - Best model based on validation accuracy
- `final_model.pt` - Model at end of training

## 🐛 Troubleshooting Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'src'"
**Solution:**
```cmd
# Make sure you're in the project root directory
cd "D:\Notes and Projects\Project\AI\VIVEK\Vigilant-Intelligent-Verifier-for-Ethical-Kids-content"
```

### Issue 2: "CUDA out of memory"
**Solution:**
1. Edit the config file (e.g., `configs/train_text_bert.yaml`)
2. Reduce `batch_size` from 16 to 8 or 4
3. Add `gradient_clip: 1.0` under training section

### Issue 3: "FileNotFoundError: data/captions.csv"
**Solution:**
- Create the data files as shown in Step 9
- Or update paths in config file

### Issue 4: "FFmpeg not found" (for video/audio models)
**Solution:**
- Install FFmpeg and add to PATH
- Or stick to text models for now

### Issue 5: Slow training on CPU
**Solution:**
- This is normal for deep learning models
- Text models are fastest (BERT ~30min, LSTM ~5min)
- Consider using GPU or cloud services for faster training

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Virtual environment activated
- [ ] PyTorch installed and working
- [ ] Data files created (captions.csv, labels.csv)
- [ ] Training script runs without errors
- [ ] Model checkpoints are being saved
- [ ] Inference script works

## 📚 Next Steps

1. **Experiment with different models:**
   - Try TextLSTM (faster)
   - Try TextCNN (different architecture)
   - Try video/audio models (if you have video data)

2. **Tune hyperparameters:**
   - Edit config files
   - Adjust learning rate, batch size, epochs

3. **Add more data:**
   - Add more samples to captions.csv and labels.csv
   - Retrain for better performance

4. **Try multi-modal fusion:**
   - Combine text, video, and audio
   - Use `configs/train_multimodal.yaml`

## 💡 Quick Reference Commands

```cmd
# Activate environment
venv\Scripts\activate

# Deactivate environment
deactivate

# Train model
python train.py --config configs/train_text_bert.yaml

# Make prediction
python inference.py --model checkpoints/text_bert/best_model.pt --config configs/train_text_bert.yaml --mode text --text "Your text"

# Check GPU availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Install missing package
pip install package_name
```

## 🎉 Success!

If you see:
- ✅ Training completes without errors
- ✅ Checkpoints saved in `checkpoints/` folder
- ✅ Inference produces predictions

**Congratulations! Your deep learning project is working!** 🚀

---

**Need Help?** Check:
- `README.md` - Full documentation
- `QUICKSTART.md` - Quick start guide
- `PROJECT_SUMMARY.md` - Project overview

