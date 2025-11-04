"""
Training Infrastructure for Deep Learning Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
from typing import Dict, Optional, List


class Trainer:
    """
    Main trainer class for deep learning models
    """
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        
    def _setup_optimizer(self):
        """Setup optimizer based on config"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        learning_rate = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=learning_rate, 
                momentum=0.9, 
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'step').lower()
        
        if scheduler_type == 'step':
            step_size = self.config.get('scheduler_step_size', 10)
            gamma = self.config.get('scheduler_gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = self.config.get('scheduler_T_max', 50)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max)
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # Move batch to device
            if isinstance(batch, dict):
                # Handle different batch formats
                if 'input_ids' in batch:
                    # Text model
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask)
                    loss = self.criterion(outputs, labels)
                elif 'frames' in batch:
                    # Video model
                    frames = batch['frames'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, labels)
                elif 'spectrogram' in batch:
                    # Audio model
                    spectrogram = batch['spectrogram'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(spectrogram)
                    loss = self.criterion(outputs, labels)
                else:
                    # Multi-modal
                    raise NotImplementedError("Multi-modal training not yet implemented in trainer")
            else:
                # Simple format
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                if isinstance(batch, dict):
                    if 'input_ids' in batch:
                        # Text model
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = self.model(input_ids, attention_mask)
                        loss = self.criterion(outputs, labels)
                    elif 'frames' in batch:
                        # Video model
                        frames = batch['frames'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = self.model(frames)
                        loss = self.criterion(outputs, labels)
                    elif 'spectrogram' in batch:
                        # Audio model
                        spectrogram = batch['spectrogram'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = self.model(spectrogram)
                        loss = self.criterion(outputs, labels)
                    else:
                        raise NotImplementedError("Multi-modal validation not yet implemented")
                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs[:, 1])  # Probability of positive class
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        try:
            epoch_auc = roc_auc_score(all_labels, all_probs)
        except:
            epoch_auc = 0.0
        
        return epoch_loss, epoch_acc, epoch_f1, epoch_auc
    
    def train(self, num_epochs):
        """Train the model for multiple epochs"""
        print(f"\n{'='*80}")
        print(f"🚀 Starting Training for {num_epochs} epochs")
        print(f"{'='*80}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1, val_auc = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            
            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f"✅ New best model! Val Acc: {val_acc:.4f}")
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pt")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n✅ Loaded best model with Val Acc: {self.best_val_acc:.4f}")
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }, checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint['best_val_acc']
        
        print(f"✅ Checkpoint loaded: {checkpoint_path}")
        return checkpoint['epoch']

