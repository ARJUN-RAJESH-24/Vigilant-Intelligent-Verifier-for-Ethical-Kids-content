#!/bin/bash

# Setup script for VIVEK project

echo "=========================================="
echo "VIVEK Project Setup"
echo "=========================================="

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/videos
mkdir -p data/images
mkdir -p features
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Download BERT tokenizer (if needed)
echo "Downloading BERT tokenizer..."
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To train a model:"
echo "  python train.py --config configs/train_text_lstm.yaml"
echo "=========================================="

