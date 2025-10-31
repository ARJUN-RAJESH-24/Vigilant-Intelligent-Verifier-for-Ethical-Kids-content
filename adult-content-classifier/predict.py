"""
predict.py - Make predictions on new video content

Usage:
    python predict.py --video path/to/video.mp4 --caption "Video description"
    python predict.py --batch predictions.csv
"""

import argparse
import joblib
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Import feature extraction functions
sys.path.append('scripts')
from extract_text_features import extract_text_features, clean_text
from extract_audio_features import extract_audio_features
from extract_video_features import extract_video_features

class ContentClassifier:
    """Adult content classifier using trained ML models."""
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize classifier with trained model and scaler.
        
        Args:
            model_path: Path to trained model (.pkl file)
            scaler_path: Path to fitted scaler (.pkl file)
        """
        # Default paths
        if model_path is None:
            model_path = "models/trained_models/RandomForest.pkl"
        if scaler_path is None:
            scaler_path = "models/scaler.pkl"
        
        # Load model and scaler
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded model: {Path(model_path).name}")
            print(f"✅ Loaded scaler: {Path(scaler_path).name}")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Please train models first by running: python scripts/train_models.py")
            sys.exit(1)
        
        # Load feature columns from training
        try:
            # Try to load sample features to get column order
            text_df = pd.read_csv("features/text_features.csv")
            audio_df = pd.read_csv("features/audio_features.csv")
            video_df = pd.read_csv("features/video_features.csv")
            
            sample = text_df.merge(audio_df, on='id', how='left') \
                           .merge(video_df, on='id', how='left')
            self.feature_columns = [col for col in sample.columns if col != 'id']
            print(f"✅ Loaded feature schema: {len(self.feature_columns)} features")
        except Exception as e:
            print(f"⚠️  Could not load feature schema: {e}")
            self.feature_columns = None
    
    def extract_features(self, video_path=None, caption=None):
        """
        Extract features from video and caption.
        
        Args:
            video_path: Path to video file
            caption: Text caption/description
            
        Returns:
            DataFrame with extracted features
        """
        features = {}
        
        # Extract text features
        if caption:
            cleaned = clean_text(caption)
            text_feats = extract_text_features(cleaned)
            features.update(text_feats)
        else:
            print("⚠️  No caption provided, using zeros for text features")
            # Add zero features for text
            features.update({
                'char_count': 0, 'word_count': 0, 'avg_word_length': 0,
                'sentiment_polarity': 0, 'sentiment_subjectivity': 0,
                'adult_keywords_count': 0, 'violence_keywords_count': 0,
                'uppercase_ratio': 0, 'punctuation_count': 0,
                'exclamation_count': 0, 'question_count': 0, 'number_count': 0
            })
        
        # Extract audio features
        if video_path and os.path.exists(video_path):
            audio_feats = extract_audio_features(video_path)
            if audio_feats:
                features.update(audio_feats)
        else:
            print("⚠️  Video not found, using zeros for audio features")
        
        # Extract video features
        if video_path and os.path.exists(video_path):
            video_feats = extract_video_features(video_path)
            if video_feats:
                features.update(video_feats)
        else:
            print("⚠️  Video not found, using zeros for video features")
        
        # Create DataFrame
        features_df = pd.DataFrame([features])
        
        # Align with training features
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[self.feature_columns]
        
        return features_df
    
    def predict(self, video_path=None, caption=None):
        """
        Make prediction on video content.
        
        Args:
            video_path: Path to video file
            caption: Text caption/description
            
        Returns:
            dict with prediction results
        """
        print("\n" + "="*80)
        print("🔍 ANALYZING CONTENT")
        print("="*80)
        
        if video_path:
            print(f"Video: {video_path}")
        if caption:
            print(f"Caption: {caption[:100]}...")
        
        # Extract features
        print("\n📊 Extracting features...")
        features = self.extract_features(video_path, caption)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction]
        else:
            probabilities = None
            confidence = None
        
        # Prepare result
        result = {
            'prediction': 'Adult' if prediction == 1 else 'Safe',
            'prediction_label': int(prediction),
            'confidence': float(confidence) if confidence else None,
            'safe_probability': float(probabilities[0]) if probabilities is not None else None,
            'adult_probability': float(probabilities[1]) if probabilities is not None else None
        }
        
        # Print result
        print("\n" + "="*80)
        print("📊 PREDICTION RESULT")
        print("="*80)
        
        emoji = "🟢" if result['prediction'] == 'Safe' else "🔴"
        print(f"\n{emoji} Prediction: {result['prediction']}")
        
        if result['confidence']:
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"\n   Probabilities:")
            print(f"   - Safe:  {result['safe_probability']:.2%}")
            print(f"   - Adult: {result['adult_probability']:.2%}")
        
        return result
    
    def predict_batch(self, input_csv, output_csv=None):
        """
        Make predictions on batch of videos.
        
        Args:
            input_csv: CSV with columns: id, video_path, caption
            output_csv: Path to save results (optional)
            
        Returns:
            DataFrame with predictions
        """
        print("\n" + "="*80)
        print("📊 BATCH PREDICTION")
        print("="*80)
        
        # Load input
        df = pd.read_csv(input_csv)
        print(f"\n✅ Loaded {len(df)} samples from {input_csv}")
        
        required_cols = ['id', 'caption']
        if not all(col in df.columns for col in required_cols):
            print(f"❌ Error: CSV must contain columns: {required_cols}")
            sys.exit(1)
        
        # Add video_path column if not exists
        if 'video_path' not in df.columns:
            df['video_path'] = None
        
        # Make predictions
        results = []
        
        print("\n🔄 Processing samples...")
        for idx, row in df.iterrows():
            print(f"\n[{idx+1}/{len(df)}] Processing {row['id']}...")
            
            result = self.predict(
                video_path=row.get('video_path'),
                caption=row['caption']
            )
            
            result['id'] = row['id']
            results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Merge with original data
        output_df = df.merge(results_df, on='id', how='left')
        
        # Save results
        if output_csv is None:
            output_csv = input_csv.replace('.csv', '_predictions.csv')
        
        output_df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to: {output_csv}")
        
        # Summary
        print("\n" + "="*80)
        print("📊 BATCH PREDICTION SUMMARY")
        print("="*80)
        
        safe_count = (output_df['prediction'] == 'Safe').sum()
        adult_count = (output_df['prediction'] == 'Adult').sum()
        
        print(f"\nTotal samples: {len(output_df)}")
        print(f"🟢 Safe:  {safe_count} ({safe_count/len(output_df)*100:.1f}%)")
        print(f"🔴 Adult: {adult_count} ({adult_count/len(output_df)*100:.1f}%)")
        
        return output_df


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Adult Content Classifier - Make predictions on video content',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--caption', type=str, help='Video caption/description')
    parser.add_argument('--batch', type=str, help='Path to CSV file for batch prediction')
    parser.add_argument('--model', type=str, default='models/trained_models/RandomForest.pkl',
                       help='Path to trained model (default: RandomForest)')
    parser.add_argument('--output', type=str, help='Output CSV path for batch predictions')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = ContentClassifier(model_path=args.model)
    
    # Batch prediction
    if args.batch:
        classifier.predict_batch(args.batch, args.output)
    
    # Single prediction
    elif args.video or args.caption:
        result = classifier.predict(video_path=args.video, caption=args.caption)
    
    else:
        print("❌ Error: Please provide either --video/--caption or --batch")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()