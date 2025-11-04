"""
Deep Learning Models for Text Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TextLSTM(nn.Module):
    """
    LSTM-based text classifier for content moderation
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, 
                 num_classes=2, dropout=0.3, pretrained_embeddings=None):
        super(TextLSTM, self).__init__()
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state from both directions
        forward_hidden = hidden[-2, :, :]  # Last forward layer
        backward_hidden = hidden[-1, :, :]  # Last backward layer
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Classification
        out = self.dropout(combined_hidden)
        out = F.relu(self.fc(out))
        out = self.dropout(out)
        out = self.classifier(out)
        
        return out


class TextCNN(nn.Module):
    """
    CNN-based text classifier using multiple filter sizes
    """
    
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, 
                 filter_sizes=[3, 4, 5], num_classes=2, dropout=0.3,
                 pretrained_embeddings=None):
        super(TextCNN, self).__init__()
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_seq_length)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate all filter outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Classification
        out = self.dropout(concatenated)
        out = self.fc(out)
        
        return out


class TextBERT(nn.Module):
    """
    BERT-based text classifier using HuggingFace Transformers
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        super(TextBERT, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Get BERT hidden size
        hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        
        # Classification
        out = self.dropout(pooled_output)
        out = self.classifier(out)
        
        return out
    
    def get_tokenizer(self):
        """Get the corresponding tokenizer for this model"""
        return AutoTokenizer.from_pretrained('bert-base-uncased' if 'bert-base' in str(type(self.bert)) else 'distilbert-base-uncased')

