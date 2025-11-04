"""
Deep Learning Models for Video Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VideoCNN(nn.Module):
    """
    CNN-based video classifier using frame-level features
    """
    
    def __init__(self, num_frames=16, num_classes=2, dropout=0.3):
        super(VideoCNN, self).__init__()
        
        # Feature extractor (pretrained ResNet18 backbone)
        resnet = models.resnet18(pretrained=True)
        # Remove the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet18
        feature_dim = 512
        
        # Temporal modeling (simple pooling or LSTM)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, 256)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.size()
        
        # Reshape to process all frames
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features for each frame
        features = self.feature_extractor(x)  # (batch_size * num_frames, feature_dim, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch_size * num_frames, feature_dim)
        
        # Reshape back to temporal structure
        features = features.view(batch_size, num_frames, -1)  # (batch_size, num_frames, feature_dim)
        
        # Temporal pooling
        features = features.permute(0, 2, 1)  # (batch_size, feature_dim, num_frames)
        pooled = self.temporal_pool(features)  # (batch_size, feature_dim, 1)
        pooled = pooled.squeeze(-1)  # (batch_size, feature_dim)
        
        # Classification
        out = self.dropout(pooled)
        out = F.relu(self.fc(out))
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out


class VideoResNet(nn.Module):
    """
    ResNet-based video classifier with 3D convolutions
    """
    
    def __init__(self, num_classes=2, dropout=0.3, model_name='resnet18'):
        super(VideoResNet, self).__init__()
        
        # Use 2D ResNet as feature extractor
        if model_name == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        elif model_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        else:
            resnet = models.resnet18(pretrained=True)
        
        # Remove the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get feature dimension
        if 'resnet50' in model_name:
            feature_dim = 2048
        else:
            feature_dim = 512
        
        # Temporal LSTM for sequence modeling
        self.lstm = nn.LSTM(feature_dim, 256, num_layers=2, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256 * 2, 256)  # *2 for bidirectional
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.size()
        
        # Process each frame
        frame_features = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]  # (batch_size, channels, height, width)
            features = self.feature_extractor(frame)  # (batch_size, feature_dim, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (batch_size, feature_dim)
            frame_features.append(features)
        
        # Stack frame features
        sequence = torch.stack(frame_features, dim=1)  # (batch_size, num_frames, feature_dim)
        
        # LSTM for temporal modeling
        lstm_out, (hidden, cell) = self.lstm(sequence)
        
        # Use last hidden state from both directions
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Classification
        out = self.dropout(combined_hidden)
        out = F.relu(self.fc(out))
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out


class VideoTransformer(nn.Module):
    """
    Transformer-based video classifier using frame-level features
    """
    
    def __init__(self, num_frames=16, num_classes=2, dropout=0.3):
        super(VideoTransformer, self).__init__()
        
        # Feature extractor (pretrained ResNet18)
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 512
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, num_frames, feature_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, 256)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.size()
        
        # Extract features for each frame
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.feature_extractor(x)
        features = features.squeeze(-1).squeeze(-1)
        features = features.view(batch_size, num_frames, -1)
        
        # Add positional encoding
        features = features + self.pos_encoding
        
        # Transformer encoding
        encoded = self.transformer(features)  # (batch_size, num_frames, feature_dim)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # (batch_size, feature_dim)
        
        # Classification
        out = self.dropout(pooled)
        out = F.relu(self.fc(out))
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out

