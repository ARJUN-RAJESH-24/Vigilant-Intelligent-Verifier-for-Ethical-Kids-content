# Quick Start - First Time Setup (Copy & Paste)

## 🚀 Step-by-Step Commands (Run in Order)

### Step 1: Navigate to Project Directory
```cmd
cd "D:\Notes and Projects\Project\AI\VIVEK\Vigilant-Intelligent-Verifier-for-Ethical-Kids-content"
```

### Step 2: Create Virtual Environment
```cmd
python -m venv venv
```

### Step 3: Activate Virtual Environment
```cmd
venv\Scripts\activate
```

### Step 4: Upgrade pip
```cmd
python -m pip install --upgrade pip
```

### Step 5: Install PyTorch (Choose ONE)

**Option A: GPU Support (Recommended if you have NVIDIA GPU)**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Option B: CPU Only**
```cmd
pip install torch torchvision torchaudio
```

### Step 6: Install All Dependencies
```cmd
pip install -r requirements.txt
```

*(This will take 5-10 minutes)*

### Step 7: Create Sample Data (Optional - Skip if you have data)

```cmd
python -c "import pandas as pd; captions = pd.DataFrame({'id': ['vid001', 'vid002', 'vid003', 'vid004', 'vid005'], 'caption': ['Kids playing soccer in the park', 'Family cooking dinner together', 'Educational science experiment', 'Hot dance performance at nightclub', 'Romantic couple intimate scene']}); captions.to_csv('data/captions.csv', index=False); labels = pd.DataFrame({'id': ['vid001', 'vid002', 'vid003', 'vid004', 'vid005'], 'label': [0, 0, 0, 1, 1]}); labels.to_csv('data/labels.csv', index=False); print('✅ Sample data created')"
```

### Step 8: Train Your First Model (Text BERT)
```cmd
python train.py --config configs/train_text_bert.yaml
```

### Step 9: Make Your First Prediction
```cmd
python inference.py --model checkpoints/text_bert/best_model.pt --config configs/train_text_bert.yaml --mode text --text "Kids playing soccer in the park"
```

---

## ✅ OR: Use Automated Setup Script

Just run:
```cmd
setup_project.bat
```

Then manually:
```cmd
venv\Scripts\activate
python train.py --config configs/train_text_bert.yaml
```

---

## 🔧 Troubleshooting

### If "ModuleNotFoundError: No module named 'src'"
Make sure you're in the project root directory:
```cmd
cd "D:\Notes and Projects\Project\AI\VIVEK\Vigilant-Intelligent-Verifier-for-Ethical-Kids-content"
```

### If "CUDA out of memory"
Edit `configs/train_text_bert.yaml` and change:
```yaml
data:
  batch_size: 8  # Change from 16 to 8 or 4
```

### If "FileNotFoundError: data/captions.csv"
Run Step 7 to create sample data, or create your own data files.

---

## 📝 What Each Step Does

1. **Navigate**: Goes to project folder
2. **Create venv**: Creates isolated Python environment
3. **Activate**: Activates the environment
4. **Upgrade pip**: Ensures latest package manager
5. **Install PyTorch**: Core deep learning framework
6. **Install requirements**: All other dependencies
7. **Create data**: Sample data for testing
8. **Train**: Trains a text classification model
9. **Predict**: Tests the trained model

---

## ⚡ Quick Test (After Setup)

```cmd
# Check if everything is installed
python -c "import torch; import transformers; print('✅ PyTorch:', torch.__version__); print('✅ Transformers:', transformers.__version__)"

# Check GPU (if available)
python -c "import torch; print('✅ CUDA available:', torch.cuda.is_available())"
```

---

**Need more details?** See `FIRST_TIME_SETUP.md` for comprehensive guide.

