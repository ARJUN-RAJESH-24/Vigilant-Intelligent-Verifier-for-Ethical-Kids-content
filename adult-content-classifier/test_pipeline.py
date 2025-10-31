"""
test_pipeline.py - Automated testing for the complete pipeline

This script tests all components of the adult content classifier:
- Feature extraction (text, audio, video)
- Model training
- Predictions
- Error handling

Run this to verify your installation and setup.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Test results storage
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def log_test(name, status, message=""):
    """Log test result."""
    if status == 'pass':
        print(f"  ✅ {name}")
        test_results['passed'].append(name)
    elif status == 'fail':
        print(f"  ❌ {name}: {message}")
        test_results['failed'].append((name, message))
    elif status == 'warn':
        print(f"  ⚠️  {name}: {message}")
        test_results['warnings'].append((name, message))

def print_section(title):
    """Print section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

# ============================================================================
# TEST 1: DEPENDENCY CHECKS
# ============================================================================
print_section("1️⃣  TESTING DEPENDENCIES")

try:
    import pandas
    log_test("pandas", "pass")
except ImportError as e:
    log_test("pandas", "fail", str(e))

try:
    import numpy
    log_test("numpy", "pass")
except ImportError as e:
    log_test("numpy", "fail", str(e))

try:
    import sklearn
    log_test("scikit-learn", "pass")
except ImportError as e:
    log_test("scikit-learn", "fail", str(e))

try:
    import librosa
    log_test("librosa (audio)", "pass")
except ImportError as e:
    log_test("librosa (audio)", "fail", str(e))

try:
    import cv2
    log_test("opencv-python (video)", "pass")
except ImportError as e:
    log_test("opencv-python (video)", "fail", str(e))

try:
    from textblob import TextBlob
    log_test("textblob (NLP)", "pass")
except ImportError as e:
    log_test("textblob (NLP)", "fail", str(e))

try:
    import joblib
    log_test("joblib", "pass")
except ImportError as e:
    log_test("joblib", "fail", str(e))

try:
    import matplotlib
    log_test("matplotlib", "pass")
except ImportError as e:
    log_test("matplotlib", "fail", str(e))

try:
    import seaborn
    log_test("seaborn", "pass")
except ImportError as e:
    log_test("seaborn", "fail", str(e))

# ============================================================================
# TEST 2: DIRECTORY STRUCTURE
# ============================================================================
print_section("2️⃣  TESTING DIRECTORY STRUCTURE")

required_dirs = [
    "data", "data/videos", "features", 
    "models", "models/trained_models", "models/results",
    "scripts", "notebooks"
]

for directory in required_dirs:
    if os.path.exists(directory):
        log_test(f"Directory: {directory}", "pass")
    else:
        log_test(f"Directory: {directory}", "warn", "Does not exist, will be created")
        os.makedirs(directory, exist_ok=True)

# ============================================================================
# TEST 3: SCRIPT FILES
# ============================================================================
print_section("3️⃣  TESTING SCRIPT FILES")

required_scripts = [
    "scripts/extract_text_features.py",
    "scripts/extract_audio_features.py",
    "scripts/extract_video_features.py",
    "scripts/train_models.py"
]

for script in required_scripts:
    if os.path.exists(script):
        log_test(f"Script: {script}", "pass")
    else:
        log_test(f"Script: {script}", "fail", "Missing")

# ============================================================================
# TEST 4: TEXT FEATURE EXTRACTION
# ============================================================================
print_section("4️⃣  TESTING TEXT FEATURE EXTRACTION")

try:
    sys.path.append('scripts')
    from extract_text_features import extract_text_features, clean_text
    
    # Test cleaning
    test_text = "Check out this video! http://example.com @user #hashtag"
    cleaned = clean_text(test_text)
    if len(cleaned) > 0 and "http" not in cleaned:
        log_test("Text cleaning", "pass")
    else:
        log_test("Text cleaning", "fail", "Cleaning not working properly")
    
    # Test feature extraction
    features = extract_text_features("This is a test caption for analysis")
    if isinstance(features, dict) and len(features) > 0:
        log_test("Text feature extraction", "pass")
        log_test(f"  Extracted {len(features)} features", "pass")
    else:
        log_test("Text feature extraction", "fail", "No features extracted")
    
except Exception as e:
    log_test("Text feature extraction", "fail", str(e))

# ============================================================================
# TEST 5: SAMPLE DATA CREATION
# ============================================================================
print_section("5️⃣  TESTING SAMPLE DATA CREATION")

try:
    # Create sample captions
    sample_captions = pd.DataFrame({
        'id': ['test001', 'test002', 'test003', 'test004'],
        'caption': [
            'Family vacation at the beach',
            'Hot dance performance at club',
            'Cooking tutorial for dinner',
            'Romantic kiss scene from movie'
        ]
    })
    sample_captions.to_csv('data/captions.csv', index=False)
    log_test("Create sample captions", "pass")
    
    # Create sample labels
    sample_labels = pd.DataFrame({
        'id': ['test001', 'test002', 'test003', 'test004'],
        'label': [0, 1, 0, 1]
    })
    sample_labels.to_csv('data/labels.csv', index=False)
    log_test("Create sample labels", "pass")
    
except Exception as e:
    log_test("Sample data creation", "fail", str(e))

# ============================================================================
# TEST 6: FEATURE EXTRACTION PIPELINE
# ============================================================================
print_section("6️⃣  TESTING FEATURE EXTRACTION PIPELINE")

# Test text features
try:
    if os.path.exists('data/captions.csv'):
        from extract_text_features import main as extract_text_main
        extract_text_main()
        
        if os.path.exists('features/text_features.csv'):
            text_df = pd.read_csv('features/text_features.csv')
            log_test(f"Text features extracted: {len(text_df)} samples", "pass")
        else:
            log_test("Text features file", "fail", "Not created")
    else:
        log_test("Text feature extraction", "warn", "No captions file")
except Exception as e:
    log_test("Text feature extraction pipeline", "fail", str(e))

# Audio features (will likely fail without videos)
if os.listdir('data/videos'):
    log_test("Audio/Video feature extraction", "warn", 
             "Videos present but not tested (requires real files)")
else:
    log_test("Audio/Video feature extraction", "warn", 
             "No videos found - add videos to data/videos/ for full testing")

# ============================================================================
# TEST 7: MODEL TRAINING (MINIMAL TEST)
# ============================================================================
print_section("7️⃣  TESTING MODEL TRAINING")

try:
    # Check if we can import sklearn models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Create minimal synthetic dataset
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)
    
    # Test scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    log_test("Feature scaling", "pass")
    
    # Test training
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_scaled, y_test)
    predictions = model.predict(X_scaled)
    log_test("Model training (RandomForest)", "pass")
    
    model2 = LogisticRegression(max_iter=100)
    model2.fit(X_scaled, y_test)
    log_test("Model training (LogisticRegression)", "pass")
    
except Exception as e:
    log_test("Model training test", "fail", str(e))

# ============================================================================
# TEST 8: MODEL PERSISTENCE
# ============================================================================
print_section("8️⃣  TESTING MODEL PERSISTENCE")

try:
    import joblib
    
    # Save model
    test_model = RandomForestClassifier(n_estimators=10)
    test_model.fit(X_test, y_test)
    joblib.dump(test_model, 'models/trained_models/test_model.pkl')
    log_test("Model saving", "pass")
    
    # Load model
    loaded_model = joblib.load('models/trained_models/test_model.pkl')
    test_pred = loaded_model.predict(X_test)
    log_test("Model loading", "pass")
    
    # Clean up
    if os.path.exists('models/trained_models/test_model.pkl'):
        os.remove('models/trained_models/test_model.pkl')
    
except Exception as e:
    log_test("Model persistence", "fail", str(e))

# ============================================================================
# TEST 9: PREDICTION FUNCTIONALITY
# ============================================================================
print_section("9️⃣  TESTING PREDICTION FUNCTIONALITY")

try:
    # Test basic prediction
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)
    model.fit(X_train, y_train)
    
    X_new = np.random.rand(1, 10)
    prediction = model.predict(X_new)
    log_test("Single prediction", "pass")
    
    # Test probability prediction
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_new)
        log_test("Probability prediction", "pass")
    
except Exception as e:
    log_test("Prediction functionality", "fail", str(e))

# ============================================================================
# TEST 10: ERROR HANDLING
# ============================================================================
print_section("🔟 TESTING ERROR HANDLING")

try:
    # Test empty text
    features = extract_text_features("")
    if features['char_count'] == 0:
        log_test("Empty text handling", "pass")
    else:
        log_test("Empty text handling", "fail", "Did not handle empty text")
    
    # Test None text
    features = extract_text_features(None)
    log_test("None text handling", "pass")
    
except Exception as e:
    log_test("Error handling", "fail", str(e))

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print_section("📊 TEST SUMMARY")

total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['warnings'])
passed = len(test_results['passed'])
failed = len(test_results['failed'])
warnings_count = len(test_results['warnings'])

print(f"\n  Total Tests: {total_tests}")
print(f"  ✅ Passed:   {passed}")
print(f"  ❌ Failed:   {failed}")
print(f"  ⚠️  Warnings: {warnings_count}")

if failed > 0:
    print(f"\n  ❌ Failed Tests:")
    for name, message in test_results['failed']:
        print(f"     - {name}: {message}")

if warnings_count > 0:
    print(f"\n  ⚠️  Warnings:")
    for name, message in test_results['warnings']:
        print(f"     - {name}: {message}")

# Overall result
print(f"\n{'='*80}")
if failed == 0:
    print("  ✅ ALL CRITICAL TESTS PASSED!")
    print("  Your installation is ready to use.")
    if warnings_count > 0:
        print(f"  Note: {warnings_count} warning(s) - review above for details")
else:
    print("  ❌ SOME TESTS FAILED!")
    print("  Please fix the issues above before proceeding.")
print(f"{'='*80}\n")

# Save test report
report = {
    'total_tests': total_tests,
    'passed': passed,
    'failed': failed,
    'warnings': warnings_count,
    'passed_tests': test_results['passed'],
    'failed_tests': [{'name': n, 'message': m} for n, m in test_results['failed']],
    'warning_tests': [{'name': n, 'message': m} for n, m in test_results['warnings']]
}

import json
with open('test_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"📄 Detailed test report saved to: test_report.json\n")

sys.exit(0 if failed == 0 else 1)