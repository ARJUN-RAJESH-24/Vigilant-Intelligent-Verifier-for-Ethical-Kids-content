"""
Deep Learning Models for Audio Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    """
    CNN-based audio classifier using spectrogram features
    """
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(AudioCNN, self).__init__()
        
        # Convolutional layers for spectrogram processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, freq_bins, time_frames) - spectrogram
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class AudioTransformer(nn.Module):
    """
    Transformer-based audio classifier using spectrogram features
    """
    
    def __init__(self, input_dim=128, num_classes=2, dropout=0.3, num_heads=8, num_layers=4):
        super(AudioTransformer, self).__init__()
        
        # Linear projection for input
        self.input_projection = nn.Linear(input_dim, 256)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, 256))  # Max sequence length
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, time_frames, freq_bins) - flattened spectrogram
        # If input is 2D spectrogram, flatten it
        if len(x.shape) == 3 and x.shape[1] != x.shape[2]:  # (batch, freq, time)
            # Take mean over frequency dimension
            x = x.mean(dim=1)  # (batch, time)
            x = x.unsqueeze(-1)  # (batch, time, 1)
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch_size, time_frames, 256)
        
        # Add positional encoding (truncate if needed)
        seq_len = x.size(1)
        pos_enc = self.pos_encoding[:, :seq_len, :]
        x = x + pos_enc
        
        # Transformer encoding
        encoded = self.transformer(x)  # (batch_size, time_frames, 256)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # (batch_size, 256)
        
        # Classification
        out = self.dropout(pooled)
        out = F.relu(self.fc(out))
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out

