"""
config.py - Centralized configuration for adult content classifier

This file contains all configurable parameters for the project.
Modify these settings to customize the behavior of feature extraction and training.
"""

import os

# ============================================================================
# DATA PATHS
# ============================================================================
DATA_DIR = "data"
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CAPTIONS_PATH = os.path.join(DATA_DIR, "captions.csv")
LABELS_PATH = os.path.join(DATA_DIR, "labels.csv")

# ============================================================================
# FEATURE PATHS
# ============================================================================
FEATURES_DIR = "features"
TEXT_FEATURES_PATH = os.path.join(FEATURES_DIR, "text_features.csv")
AUDIO_FEATURES_PATH = os.path.join(FEATURES_DIR, "audio_features.csv")
VIDEO_FEATURES_PATH = os.path.join(FEATURES_DIR, "video_features.csv")

# ============================================================================
# MODEL PATHS
# ============================================================================
MODELS_DIR = "models"
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")
RESULTS_DIR = os.path.join(MODELS_DIR, "results")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.csv")

# ============================================================================
# AUDIO FEATURE EXTRACTION SETTINGS
# ============================================================================
AUDIO_CONFIG = {
    'sample_rate': 22050,           # Sample rate for audio processing
    'duration': 30,                 # Max duration to process (seconds)
    'mono': True,                   # Convert to mono
    'n_mfcc': 13,                   # Number of MFCC coefficients
    'hop_length': 512,              # Hop length for feature extraction
    'n_fft': 2048,                  # FFT window size
}

# ============================================================================
# VIDEO FEATURE EXTRACTION SETTINGS
# ============================================================================
VIDEO_CONFIG = {
    'max_frames': 300,              # Maximum frames to process
    'sample_interval': 5,           # Sample every N-th frame
    'resize_width': None,           # Resize width (None = no resize)
    'resize_height': None,          # Resize height (None = no resize)
    'skin_detection': True,         # Enable skin detection
    'edge_detection': True,         # Enable edge detection
    'color_analysis': True,         # Enable color analysis
    'scene_cut_threshold': 0.7,     # Threshold for scene cut detection (0-1)
}

# YCrCb color space thresholds for skin detection
SKIN_DETECTION_CONFIG = {
    'lower_bound': [0, 133, 77],    # Lower YCrCb threshold
    'upper_bound': [255, 173, 127], # Upper YCrCb threshold
}

# ============================================================================
# TEXT FEATURE EXTRACTION SETTINGS
# ============================================================================
TEXT_CONFIG = {
    'remove_urls': True,            # Remove URLs from text
    'remove_mentions': True,        # Remove @mentions
    'remove_hashtags': True,        # Remove #hashtags
    'lowercase': True,              # Convert to lowercase
    'remove_special_chars': True,   # Remove special characters
}

# Keywords for content detection
ADULT_KEYWORDS = [
    "hot", "sexy", "nude", "naked", "kiss", "bed", "lust", "erotic", "nsfw",
    "sex", "porn", "xxx", "adult", "intimate", "sensual", "seductive",
    "provocative", "explicit", "mature", "bikini", "lingerie", "bedroom",
    "romantic", "passion", "desire"
]

VIOLENCE_KEYWORDS = [
    "kill", "blood", "murder", "death", "violent", "fight", "attack",
    "weapon", "gun", "knife", "war", "combat", "shoot", "stab", "gore",
    "brutal", "torture", "assault"
]

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================
TRAINING_CONFIG = {
    'test_size': 0.2,               # Test set size (0.2 = 20%)
    'random_state': 42,             # Random seed for reproducibility
    'cv_folds': 5,                  # Cross-validation folds
    'stratify': True,               # Stratified split
    'scale_features': True,         # Scale features using StandardScaler
}

# ============================================================================
# HYPERPARAMETER GRIDS FOR GRID SEARCH
# ============================================================================
HYPERPARAMETER_GRIDS = {
    'LogisticRegression': {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000]
    },
    'SVM': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'RandomForest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    },
    'NaiveBayes': {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    },
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'DecisionTree': {
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
EVALUATION_CONFIG = {
    'save_confusion_matrix': True,  # Save confusion matrix plots
    'save_roc_curve': True,         # Save ROC curve plots
    'save_pr_curve': True,          # Save precision-recall curve plots
    'plot_dpi': 300,                # DPI for saved plots
    'plot_format': 'png',           # Format for saved plots (png, pdf, svg)
}

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================
PREDICTION_CONFIG = {
    'default_model': 'RandomForest', # Default model for predictions
    'confidence_threshold': 0.5,     # Threshold for binary classification
    'batch_size': 100,               # Batch size for batch predictions
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
LOGGING_CONFIG = {
    'log_level': 'INFO',            # Logging level (DEBUG, INFO, WARNING, ERROR)
    'log_file': 'classifier.log',   # Log file path
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
PERFORMANCE_CONFIG = {
    'n_jobs': -1,                   # Number of parallel jobs (-1 = all cores)
    'verbose': 1,                   # Verbosity level (0, 1, 2)
    'cache_features': True,         # Cache extracted features
}

# ============================================================================
# FILE FORMATS
# ============================================================================
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.aac', '.flac', '.ogg']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, VIDEO_DIR, IMAGE_DIR, FEATURES_DIR,
        MODELS_DIR, TRAINED_MODELS_DIR, RESULTS_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ All directories created/verified")

def get_config_summary():
    """Get a summary of current configuration."""
    summary = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║           ADULT CONTENT CLASSIFIER - CONFIGURATION          ║
    ╚══════════════════════════════════════════════════════════════╝
    
    📁 Paths:
       Data Directory:     {DATA_DIR}
       Features Directory: {FEATURES_DIR}
       Models Directory:   {MODELS_DIR}
    
    🎵 Audio Settings:
       Sample Rate:        {AUDIO_CONFIG['sample_rate']} Hz
       Duration:           {AUDIO_CONFIG['duration']} seconds
       MFCCs:              {AUDIO_CONFIG['n_mfcc']} coefficients
    
    🎬 Video Settings:
       Max Frames:         {VIDEO_CONFIG['max_frames']}
       Sample Interval:    {VIDEO_CONFIG['sample_interval']}
       Skin Detection:     {VIDEO_CONFIG['skin_detection']}
    
    📝 Text Settings:
       Adult Keywords:     {len(ADULT_KEYWORDS)}
       Violence Keywords:  {len(VIOLENCE_KEYWORDS)}
    
    🤖 Training Settings:
       Test Size:          {TRAINING_CONFIG['test_size']*100}%
       CV Folds:           {TRAINING_CONFIG['cv_folds']}
       Random State:       {TRAINING_CONFIG['random_state']}
    
    🎯 Models:
       {len(HYPERPARAMETER_GRIDS)} classifiers configured
    """
    return summary

if __name__ == "__main__":
    # Create directories when run directly
    create_directories()
    print(get_config_summary())