"""
Training Callbacks
"""

import torch
import numpy as np
from typing import Callable


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving
    """
    
    def __init__(self, patience=10, min_delta=0.0, mode='max', monitor='val_acc'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.best_value = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
        else:
            if self.mode == 'max':
                improved = current_value > self.best_value + self.min_delta
            else:
                improved = current_value < self.best_value - self.min_delta
            
            if improved:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """
    Model checkpoint callback to save model at specific intervals
    """
    
    def __init__(self, checkpoint_dir='checkpoints', save_best=True, monitor='val_acc', mode='max'):
        self.checkpoint_dir = checkpoint_dir
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.best_value = None
    
    def save(self, model, epoch, metrics, filename=None):
        """Save model checkpoint"""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
        
        # Save best model
        if self.save_best:
            current_value = metrics.get(self.monitor, 0)
            if self.best_value is None:
                self.best_value = current_value
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, best_path)
            else:
                if self.mode == 'max' and current_value > self.best_value:
                    self.best_value = current_value
                    best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'metrics': metrics
                    }, best_path)
                elif self.mode == 'min' and current_value < self.best_value:
                    self.best_value = current_value
                    best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'metrics': metrics
                    }, best_path)


class LearningRateScheduler:
    """
    Learning rate scheduler callback
    """
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def step(self, metric=None):
        """Step the scheduler"""
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()

