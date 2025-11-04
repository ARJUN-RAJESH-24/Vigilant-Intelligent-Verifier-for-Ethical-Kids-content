@echo off
REM Setup script for VIVEK project (Windows)

echo ==========================================
echo VIVEK Project Setup
echo ==========================================

REM Create necessary directories
echo Creating project directories...
if not exist data\videos mkdir data\videos
if not exist data\images mkdir data\images
if not exist features mkdir features
if not exist checkpoints mkdir checkpoints
if not exist logs mkdir logs
if not exist results mkdir results

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Download BERT tokenizer (if needed)
echo Downloading BERT tokenizer...
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"

echo ==========================================
echo Setup completed successfully!
echo ==========================================
echo To activate the virtual environment:
echo   venv\Scripts\activate
echo.
echo To train a model:
echo   python train.py --config configs/train_text_lstm.yaml
echo ==========================================
pause

