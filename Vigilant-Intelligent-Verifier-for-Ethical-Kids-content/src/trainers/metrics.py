"""
Metrics Tracker for Training
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class MetricsTracker:
    """
    Track and compute training metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.probabilities = []
        self.labels = []
    
    def update(self, predictions, labels, probabilities=None):
        """Update metrics with new batch"""
        self.predictions.extend(predictions)
        self.labels.extend(labels)
        if probabilities is not None:
            self.probabilities.extend(probabilities)
    
    def compute(self):
        """Compute all metrics"""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='binary', zero_division=0),
            'recall': recall_score(labels, predictions, average='binary', zero_division=0),
            'f1': f1_score(labels, predictions, average='binary', zero_division=0)
        }
        
        if len(self.probabilities) > 0:
            probabilities = np.array(self.probabilities)
            try:
                metrics['auc'] = roc_auc_score(labels, probabilities)
            except:
                metrics['auc'] = 0.0
        
        metrics['confusion_matrix'] = confusion_matrix(labels, predictions)
        
        return metrics
    
    def get_classification_report(self):
        """Get detailed classification report"""
        return classification_report(
            self.labels, 
            self.predictions,
            target_names=['Safe', 'Adult']
        )

