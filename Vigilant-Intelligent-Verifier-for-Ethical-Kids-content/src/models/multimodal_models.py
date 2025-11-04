"""
Multi-Modal Fusion Models for Content Moderation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusion(nn.Module):
    """
    Early fusion: concatenate features before classification
    """
    
    def __init__(self, text_dim, video_dim, audio_dim, num_classes=2, dropout=0.3):
        super(EarlyFusion, self).__init__()
        
        # Combine all modalities
        combined_dim = text_dim + video_dim + audio_dim
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(combined_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, text_features, video_features, audio_features):
        # Concatenate all features
        combined = torch.cat([text_features, video_features, audio_features], dim=1)
        
        # Classification
        out = self.dropout(combined)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out


class LateFusion(nn.Module):
    """
    Late fusion: classify each modality separately, then combine predictions
    """
    
    def __init__(self, text_dim, video_dim, audio_dim, num_classes=2, dropout=0.3):
        super(LateFusion, self).__init__()
        
        # Individual classifiers for each modality
        self.text_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.video_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(video_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.audio_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(audio_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_classes * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, text_features, video_features, audio_features):
        # Get predictions from each modality
        text_pred = self.text_classifier(text_features)
        video_pred = self.video_classifier(video_features)
        audio_pred = self.audio_classifier(audio_features)
        
        # Concatenate predictions
        combined_pred = torch.cat([text_pred, video_pred, audio_pred], dim=1)
        
        # Final fusion
        out = self.fusion(combined_pred)
        
        return out


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion with attention mechanism
    """
    
    def __init__(self, text_dim, video_dim, audio_dim, num_classes=2, dropout=0.3):
        super(MultiModalFusion, self).__init__()
        
        # Project all modalities to same dimension
        self.projection_dim = 256
        
        self.text_projection = nn.Linear(text_dim, self.projection_dim)
        self.video_projection = nn.Linear(video_dim, self.projection_dim)
        self.audio_projection = nn.Linear(audio_dim, self.projection_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.projection_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.projection_dim * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, text_features, video_features, audio_features):
        # Project to same dimension
        text_proj = self.text_projection(text_features).unsqueeze(1)  # (batch, 1, proj_dim)
        video_proj = self.video_projection(video_features).unsqueeze(1)  # (batch, 1, proj_dim)
        audio_proj = self.audio_projection(audio_features).unsqueeze(1)  # (batch, 1, proj_dim)
        
        # Stack modalities
        modalities = torch.cat([text_proj, video_proj, audio_proj], dim=1)  # (batch, 3, proj_dim)
        
        # Self-attention
        attended, _ = self.attention(modalities, modalities, modalities)
        
        # Flatten and concatenate
        attended_flat = attended.view(attended.size(0), -1)  # (batch, 3 * proj_dim)
        
        # Classification
        out = self.dropout(attended_flat)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out

