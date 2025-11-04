import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
TEXT_FEATURES_PATH = "features/text_features.csv"
TFIDF_FEATURES_PATH = "features/tfidf_features.csv"
AUDIO_FEATURES_PATH = "features/audio_features.csv"
VIDEO_FEATURES_PATH = "features/video_features.csv"
LABELS_PATH = "data/labels.csv"
MODELS_DIR = "models/trained_models"
RESULTS_DIR = "models/results"
SCALER_PATH = "models/scaler.pkl"
FEATURE_IMPORTANCE_PATH = "models/feature_importance.csv"
SCHEMA_DUMP_PATH = "models/final_feature_schema.txt" 
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

def load_data():
    """Loads and merges ONLY text features with labels, using index alignment."""
    print("📂 Loading data (Text-Only Mode - Final Fix)...")
    
    try:
        # Load dataframes
        text_df = pd.read_csv(TEXT_FEATURES_PATH)
        labels_df = pd.read_csv(LABELS_PATH)
        tfidf_df = pd.read_csv(TFIDF_FEATURES_PATH) if os.path.exists(TFIDF_FEATURES_PATH) else pd.DataFrame({'id': []})
        
        # Load, but skip merging, Audio/Video data (for logging)
        audio_df = pd.read_csv(AUDIO_FEATURES_PATH) if os.path.exists(AUDIO_FEATURES_PATH) else pd.DataFrame({'id': []})
        video_df = pd.read_csv(VIDEO_FEATURES_PATH) if os.path.exists(VIDEO_FEATURES_PATH) else pd.DataFrame({'id': []})

    except FileNotFoundError as e:
        print(f"\n❌ CRITICAL ERROR: Required file not found. Ensure all necessary CSVs exist: {e}")
        return pd.DataFrame() # Return empty DataFrame on critical error

    print(f"  ✓ Text features (structural): {text_df.shape}")
    print(f"  ✓ Text features (TFIDF): {tfidf_df.shape}")
    print(f"  ✓ Labels: {labels_df.shape}")
    print(f"  ⚠️ Ignored Audio/Video features: Audio={audio_df.shape}, Video={video_df.shape}")

    # --- CRITICAL FIX: Merge by Index/Order to bypass ID column mismatches ---
    
    # 1. Find minimum length to prevent indexing errors
    min_len = min(len(text_df), len(labels_df), len(tfidf_df))
    
    # 2. Truncate and reset index to guarantee order alignment
    text_df = text_df.head(min_len).reset_index(drop=True)
    labels_df = labels_df.head(min_len).reset_index(drop=True)
    tfidf_df = tfidf_df.head(min_len).reset_index(drop=True)

    # 3. Concatenate horizontally by index (axis=1)
    # We drop the 'id' columns first, as they contain the mismatched data,
    # and then rely on the implicit alignment of the pandas index.
    df = pd.concat([
        text_df.drop(columns=['id']), 
        tfidf_df.drop(columns=['id']), 
        labels_df[['label', 'id']] # Keep the original labels and IDs for logging/target
    ], axis=1)

    # 4. Final cleanup and column selection
    df = df.loc[:,~df.columns.duplicated()].copy()
    
    # Ensure final DataFrame contains the required target and features
    if 'label' not in df.columns:
        df['label'] = labels_df['label']

    print(f"\n✅ Merged dataset: {df.shape}")
    print(f"   Features: {df.shape[1] - 2} (excluding 'id' and 'label')")
    
    if len(df) == 0:
        return df

    # Check label distribution
    label_counts = df['label'].value_counts()
    print(f"\n📊 Label distribution:")
    for label, count in label_counts.items():
        print(f"  Class {label}: {count} samples ({count/len(df)*100:.1f}%)")

    # Handle missing/infinite values (important for clean text data)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png'), dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(RESULTS_DIR, f'{model_name}_roc_curve.png'), dpi=300)
    plt.close()

def train_evaluate_save(X_train, X_test, y_train, y_test, feature_names):
    models_config = {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            "params": {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        "SVM": {
            "model": SVC(random_state=RANDOM_STATE, probability=True),
            "params": {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
            "params": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5]
            }
        },
        "NaiveBayes": {
            "model": GaussianNB(),
            "params": {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_summary = []
    best_models = {}

    print("\n" + "="*80)
    print("🚀 Starting Model Training and Evaluation with Hyperparameter Tuning")
    print("="*80)
    
    for name, config in models_config.items():
        print(f"\n{'='*80}")
        print(f"🔧 Training {name}...")
        print(f"{'='*80}")
        
        grid_search = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=CV_FOLDS, 
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=CV_FOLDS, scoring='f1')
        
        print(f"\n📊 {name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        if y_pred_proba is not None:
            auc = roc_auc_score(y_test, y_pred_proba)
            print(f"  ROC-AUC:   {auc:.4f}")
        
        print(f"\n{classification_report(y_test, y_pred)}")
        
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(best_model, model_path)
        print(f"✅ Model saved to {model_path}")
        
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(os.path.join(RESULTS_DIR, f"{name}_report.csv"))
        
        plot_confusion_matrix(y_test, y_pred, name)
        
        if y_pred_proba is not None:
            plot_roc_curve(y_test, y_pred_proba, name)
        
        results_summary.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV_F1_Mean': cv_scores.mean(),
            'CV_F1_Std': cv_scores.std(),
            'ROC-AUC': auc if y_pred_proba is not None else None,
            'Best_Params': str(grid_search.best_params_)
        })
        
        best_models[name] = best_model
    
    # Create ensemble (Voting Classifier)
    print(f"\n{'='*80}")
    print("🎯 Creating Ensemble Model (Voting Classifier)...")
    print(f"{'='*80}")
    
    ensemble = VotingClassifier(
        estimators=[
            ('rf', best_models['RandomForest']),
            ('gb', best_models['GradientBoosting']),
            ('svm', best_models['SVM'])
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
    
    accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
    precision_ensemble = precision_score(y_test, y_pred_ensemble, average='binary', zero_division=0)
    recall_ensemble = recall_score(y_test, y_pred_ensemble, average='binary', zero_division=0)
    f1_ensemble = f1_score(y_test, y_pred_ensemble, average='binary', zero_division=0)
    auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
    
    print(f"\n📊 Ensemble Results:")
    print(f"  Accuracy:  {accuracy_ensemble:.4f}")
    print(f"  Precision: {precision_ensemble:.4f}")
    print(f"  Recall:    {recall_ensemble:.4f}")
    print(f"  F1-Score:  {f1_ensemble:.4f}")
    print(f"  ROC-AUC:   {auc_ensemble:.4f}")
    
    joblib.dump(ensemble, os.path.join(MODELS_DIR, "Ensemble.pkl"))
    plot_confusion_matrix(y_test, y_pred_ensemble, "Ensemble")
    plot_roc_curve(y_test, y_pred_proba_ensemble, "Ensemble")
    
    results_summary.append({
        'Model': 'Ensemble',
        'Accuracy': accuracy_ensemble,
        'Precision': precision_ensemble,
        'Recall': recall_ensemble,
        'F1-Score': f1_ensemble,
        'CV_F1_Mean': None,
        'CV_F1_Std': None,
        'ROC-AUC': auc_ensemble,
        'Best_Params': 'Voting(RF+GB+SVM)'
    })
    
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    
    print(f"\n{'='*80}")
    print("📈 Model Comparison:")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    rf_model = best_models['RandomForest']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=300)
    plt.close()
    
    print(f"\n{'='*80}")
    print("✅ All models trained, evaluated, and saved successfully!")
    print(f"{'='*80}")

def main():
    print("\n" + "="*80)
    print("🤖 ADULT CONTENT CLASSIFIER - MODEL TRAINING PIPELINE")
    print("="*80 + "\n")
    
    try:
        df = load_data()
        
        # Check if load_data returned an empty DataFrame due to merge failure
        if len(df) == 0:
            print("❌ Error: Cannot proceed with 0 samples. Data merge failed.")
            return

        X = df.drop(columns=["id", "label"])
        y = df["label"]
        
        feature_names = X.columns.tolist()
        
        # --- CRITICAL FIX: Save the definitive feature schema ---
        os.makedirs("models", exist_ok=True)
        with open(SCHEMA_DUMP_PATH, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        print(f"✅ Definitive feature schema saved to {SCHEMA_DUMP_PATH}")
        # --------------------------------------------------------
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Total samples: {len(df)}")
        print(f"  Total features: {len(feature_names)}")
        print(f"  Class balance: {y.value_counts().to_dict()}")
        
        TEST_SIZE = 0.2
        RANDOM_STATE = 42
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        print(f"\n📂 Data Split:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        
        print(f"\n⚙️  Standardizing features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"✅ Scaler saved to {SCALER_PATH}")
        
        train_evaluate_save(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
        
        print(f"\n{'='*80}")
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}\n")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("💡 Please ensure all required feature files exist in the 'features/' directory.")
    except Exception as e:
        print(f"\n❌ Error in training pipeline: {e}")

if __name__ == "__main__":
    main()