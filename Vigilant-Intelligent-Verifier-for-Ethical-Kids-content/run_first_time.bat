@echo off
REM Automated First-Time Setup Script for VIVEK Project
REM This script does everything needed for first-time setup

echo ==========================================
echo VIVEK - First Time Setup
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/8] Python found: 
python --version
echo.

REM Create directories
echo [2/8] Creating project directories...
if not exist data mkdir data
if not exist data\videos mkdir data\videos
if not exist data\images mkdir data\images
if not exist features mkdir features
if not exist checkpoints mkdir checkpoints
if not exist logs mkdir logs
if not exist results mkdir results
echo ✅ Directories created
echo.

REM Create virtual environment
echo [3/8] Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    echo ✅ Virtual environment created
)
echo.

REM Activate virtual environment
echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment activated
echo.

REM Upgrade pip
echo [5/8] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✅ pip upgraded
echo.

REM Install PyTorch
echo [6/8] Installing PyTorch...
echo ⚠️  This may take a few minutes...
echo.
echo Choose installation method:
echo [1] GPU support (CUDA 11.8) - Recommended if you have NVIDIA GPU
echo [2] CPU only - Use if no GPU or unsure
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
) else (
    echo Installing PyTorch for CPU...
    pip install torch torchvision torchaudio --quiet
)
echo ✅ PyTorch installed
echo.

REM Install requirements
echo [7/8] Installing project dependencies...
echo ⚠️  This will take 5-10 minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo ✅ Dependencies installed
echo.

REM Create sample data
echo [8/8] Creating sample data...
python -c "import pandas as pd; import os; os.makedirs('data', exist_ok=True); captions = pd.DataFrame({'id': ['vid001', 'vid002', 'vid003', 'vid004', 'vid005'], 'caption': ['Kids playing soccer in the park', 'Family cooking dinner together', 'Educational science experiment', 'Hot dance performance at nightclub', 'Romantic couple intimate scene']}); captions.to_csv('data/captions.csv', index=False); labels = pd.DataFrame({'id': ['vid001', 'vid002', 'vid003', 'vid004', 'vid005'], 'label': [0, 0, 0, 1, 1]}); labels.to_csv('data/labels.csv', index=False); print('✅ Sample data created')"
echo ✅ Sample data created
echo.

REM Verify installation
echo ==========================================
echo Verifying installation...
echo ==========================================
python -c "import torch; import transformers; print('✅ PyTorch:', torch.__version__); print('✅ Transformers:', transformers.__version__); print('✅ CUDA available:', torch.cuda.is_available())"
echo.

echo ==========================================
echo ✅ SETUP COMPLETED SUCCESSFULLY!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Keep this terminal open (virtual environment is active)
echo.
echo 2. Train your first model:
echo    python train.py --config configs/train_text_bert.yaml
echo.
echo 3. Make a prediction:
echo    python inference.py --model checkpoints/text_bert/best_model.pt --config configs/train_text_bert.yaml --mode text --text "Your text here"
echo.
echo ==========================================
echo.
echo ⚠️  Note: If you close this terminal, you'll need to activate
echo    the virtual environment again:
echo    venv\Scripts\activate
echo.
pause

