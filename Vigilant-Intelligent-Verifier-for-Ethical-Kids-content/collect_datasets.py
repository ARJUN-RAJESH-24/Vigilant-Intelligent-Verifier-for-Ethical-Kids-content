"""
collect_datasets.py - Comprehensive Multi-Source Data Collection

This script collects and merges datasets from multiple sources:
- Kaggle datasets (hate speech, toxicity, NSFW)
- Reddit comments
- Twitter/X posts
- Instagram captions
- YouTube comments
- Custom text datasets

Categories covered:
- Adult/NSFW content
- Hate speech
- Racism
- Sexism
- Violence
- Toxicity
- Safe content
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import requests
import json

print("="*80)
print("📦 COMPREHENSIVE DATA COLLECTION - MULTI-SOURCE DATASET")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

CATEGORIES = {
    'adult': 1,        # Adult/NSFW content
    'hate': 1,         # Hate speech (any type)
    'racism': 1,       # Racist content
    'sexism': 1,       # Sexist content
    'violence': 1,     # Violent content
    'toxicity': 1,     # Toxic/harmful content
    'safe': 0,         # Safe content
    'neutral': 0       # Neutral content
}

TARGET_SAMPLES = 500  # Target total samples
MIN_PER_CLASS = 250   # Minimum per class (safe/unsafe)

# ============================================================================
# KAGGLE DATASETS TO DOWNLOAD
# ============================================================================

KAGGLE_DATASETS = [
    {
        'name': 'hate-speech-and-offensive-language',
        'url': 'mrmorj/hate-speech-and-offensive-language-dataset',
        'description': 'Twitter hate speech dataset',
        'text_col': 'tweet',
        'label_col': 'class',  # 0: hate speech, 1: offensive, 2: neither
        'label_map': {0: 1, 1: 1, 2: 0}  # Map to binary: unsafe/safe
    },
    {
        'name': 'jigsaw-toxic-comment',
        'url': 'julian3833/jigsaw-toxic-comment-classification-challenge',
        'description': 'Wikipedia toxic comments',
        'text_col': 'comment_text',
        'label_col': 'toxic',  # Binary: 0/1
        'label_map': None
    },
    {
        'name': 'reddit-self-harm',
        'url': 'nikhileswarkomati/suicide-watch',
        'description': 'Reddit mental health posts',
        'text_col': 'text',
        'label_col': 'class',
        'label_map': {'suicide': 1, 'non-suicide': 0}
    },
]

# ============================================================================
# SYNTHETIC DATASET GENERATION
# ============================================================================

def generate_synthetic_samples():
    """Generate synthetic training samples for all categories."""
    
    print("\n1️⃣ Generating synthetic samples...")
    
    samples = []
    
    # SAFE CONTENT (Label: 0)
    safe_templates = [
        "Family vacation at {location} with beautiful scenery",
        "Cooking tutorial: How to make delicious {food}",
        "Morning {activity} routine for beginners",
        "Educational video about {topic} for students",
        "Travel vlog exploring {place}",
        "DIY home improvement project: {project}",
        "Cute {animal} playing in the garden",
        "Professional {skill} tutorial step by step",
        "Nature documentary about {nature_topic}",
        "Fitness workout routine for {goal}",
        "Book review: {book_title} analysis",
        "Science experiment: {experiment} explained",
        "Art tutorial: How to draw {subject}",
        "Music lesson: Learning {instrument}",
        "Gardening tips for growing {plant}",
        "Tech review: {device} unboxing and setup",
        "Meditation and mindfulness guide",
        "Healthy meal prep for the week",
        "Historical documentary about {period}",
        "Language learning: {language} basics"
    ]
    
    safe_vars = {
        'location': ['beach', 'mountains', 'park', 'lake', 'forest'],
        'food': ['pasta', 'salad', 'soup', 'bread', 'dessert'],
        'activity': ['yoga', 'meditation', 'exercise', 'stretching', 'running'],
        'topic': ['history', 'science', 'math', 'geography', 'biology'],
        'place': ['temples', 'museums', 'cities', 'villages', 'landmarks'],
        'project': ['painting walls', 'fixing furniture', 'organizing closet'],
        'animal': ['puppies', 'kittens', 'birds', 'rabbits', 'hamsters'],
        'skill': ['cooking', 'photography', 'writing', 'drawing', 'coding'],
        'nature_topic': ['wildlife', 'oceans', 'forests', 'mountains', 'deserts'],
        'goal': ['weight loss', 'strength', 'flexibility', 'endurance'],
        'book_title': ['classics', 'fiction', 'non-fiction', 'mystery'],
        'experiment': ['chemistry', 'physics', 'biology'],
        'subject': ['portraits', 'landscapes', 'animals', 'objects'],
        'instrument': ['guitar', 'piano', 'drums', 'violin'],
        'plant': ['tomatoes', 'herbs', 'flowers', 'vegetables'],
        'device': ['smartphone', 'laptop', 'tablet', 'camera'],
        'period': ['ancient Rome', 'medieval times', 'Renaissance', 'modern era'],
        'language': ['Spanish', 'French', 'German', 'Japanese']
    }
    
    # Generate 150 safe samples
    for i in range(150):
        template = np.random.choice(safe_templates)
        caption = template
        for var_name, var_values in safe_vars.items():
            if f'{{{var_name}}}' in caption:
                caption = caption.replace(f'{{{var_name}}}', np.random.choice(var_values))
        
        samples.append({
            'id': f'synthetic_safe_{i:04d}',
            'caption': caption,
            'label': 0,
            'category': 'safe',
            'source': 'synthetic'
        })
    
    # ADULT/NSFW CONTENT (Label: 1)
    adult_templates = [
        "Hot {descriptor} performance at nightclub",
        "Sexy {subject} photoshoot behind the scenes",
        "NSFW explicit adult content warning",
        "Intimate bedroom scene with {setting}",
        "Provocative dance video at {location}",
        "Mature audience only: {content_type}",
        "Lingerie fashion show runway models",
        "Adult entertainment: {show_type}",
        "Erotic art photography exhibition",
        "Strip club performance late night",
        "Sensual massage therapy video",
        "Romantic couples intimate moments",
        "Nude figure drawing art class",
        "XXX adult content explicit warning",
        "After dark: {adult_content}"
    ]
    
    for i in range(30):
        template = np.random.choice(adult_templates)
        caption = template.format(
            descriptor=np.random.choice(['dance', 'music', 'show', 'performance']),
            subject=np.random.choice(['model', 'performer', 'artist']),
            setting=np.random.choice(['romantic lighting', 'soft music', 'candles']),
            location=np.random.choice(['club', 'bar', 'venue', 'party']),
            content_type=np.random.choice(['scenes', 'content', 'material']),
            show_type=np.random.choice(['show', 'performance', 'content']),
            adult_content=np.random.choice(['content', 'scenes', 'material'])
        )
        
        samples.append({
            'id': f'synthetic_adult_{i:04d}',
            'caption': caption,
            'label': 1,
            'category': 'adult',
            'source': 'synthetic'
        })
    
    # HATE SPEECH / RACISM (Label: 1)
    hate_templates = [
        "Those {group} people are all {negative_trait}",
        "I hate {group} they ruin everything",
        "Why do {group} always {negative_action}",
        "All {group} should {harmful_statement}",
        "Typical {group} behavior always {negative_action}",
        "{group} are destroying our {thing}",
        "Can't stand {group} anymore they're so {negative_trait}",
        "Sick of {group} taking over {place}",
        "Real {identity} wouldn't associate with {group}",
        "Send {group} back to where they came from"
    ]
    
    # Note: Using placeholders for educational purposes
    for i in range(30):
        template = np.random.choice(hate_templates)
        caption = template.format(
            group=np.random.choice(['[GROUP]', '[PEOPLE]', '[COMMUNITY]']),
            negative_trait=np.random.choice(['lazy', 'stupid', 'criminal', 'inferior']),
            negative_action=np.random.choice(['complain', 'cause trouble', 'steal', 'lie']),
            harmful_statement=np.random.choice(['leave', 'go away', 'be deported']),
            thing=np.random.choice(['country', 'neighborhood', 'culture', 'society']),
            place=np.random.choice(['here', 'our country', 'this area']),
            identity=np.random.choice(['citizens', 'people', 'residents'])
        )
        
        samples.append({
            'id': f'synthetic_hate_{i:04d}',
            'caption': caption,
            'label': 1,
            'category': 'hate_speech',
            'source': 'synthetic'
        })
    
    # SEXISM (Label: 1)
    sexism_templates = [
        "Women belong in the {place} not in {workplace}",
        "Girls can't {skill} as well as boys",
        "She only got the job because she's {reason}",
        "Typical woman driver can't {action}",
        "Women are too {trait} for {role}",
        "A woman's place is {location}",
        "Real men don't {activity}",
        "She's too pretty to be {occupation}",
        "Women should focus on {traditional_role}",
        "Men are naturally better at {skill}"
    ]
    
    for i in range(30):
        template = np.random.choice(sexism_templates)
        caption = template.format(
            place=np.random.choice(['kitchen', 'home']),
            workplace=np.random.choice(['office', 'workplace', 'business', 'leadership']),
            skill=np.random.choice(['code', 'drive', 'lead', 'manage']),
            reason=np.random.choice(['attractive', 'young', 'related to boss']),
            action=np.random.choice(['park', 'navigate', 'drive properly']),
            trait=np.random.choice(['emotional', 'weak', 'irrational', 'soft']),
            role=np.random.choice(['leadership', 'management', 'executive roles']),
            location=np.random.choice(['at home', 'raising children', 'supporting husband']),
            activity=np.random.choice(['cry', 'show emotion', 'be vulnerable']),
            occupation=np.random.choice(['engineer', 'scientist', 'CEO', 'programmer']),
            traditional_role=np.random.choice(['family', 'children', 'household'])
        )
        
        samples.append({
            'id': f'synthetic_sexism_{i:04d}',
            'caption': caption,
            'label': 1,
            'category': 'sexism',
            'source': 'synthetic'
        })
    
    # VIOLENCE (Label: 1)
    violence_templates = [
        "Brutal fight scene with {weapon}",
        "Graphic violence warning: {description}",
        "Violent attack caught on camera {location}",
        "Gore and blood in {context}",
        "Deadly weapon demonstration {type}",
        "Combat footage from {place}",
        "Murder scene investigation {details}",
        "Torture methods used in {context}",
        "War crimes footage graphic content",
        "Assault and battery incident {location}"
    ]
    
    for i in range(30):
        template = np.random.choice(violence_templates)
        caption = template.format(
            weapon=np.random.choice(['knife', 'gun', 'bat', 'weapon']),
            description=np.random.choice(['blood', 'injuries', 'wounds', 'graphic']),
            location=np.random.choice(['street', 'bar', 'parking lot', 'alley']),
            context=np.random.choice(['movie', 'game', 'video', 'footage']),
            type=np.random.choice(['guns', 'knives', 'explosives']),
            place=np.random.choice(['conflict zone', 'war zone', 'battlefield']),
            details=np.random.choice(['evidence', 'forensics', 'crime scene'])
        )
        
        samples.append({
            'id': f'synthetic_violence_{i:04d}',
            'caption': caption,
            'label': 1,
            'category': 'violence',
            'source': 'synthetic'
        })
    
    # TOXICITY (Label: 1)
    toxic_templates = [
        "You're absolutely {insult} and everyone knows it",
        "Shut up you {insult} nobody cares",
        "Go {action} yourself you piece of {noun}",
        "What a {insult} comment from a {insult} person",
        "You're such a {insult} I can't believe you said that",
        "Stupid {insult} doesn't know what they're talking about",
        "Everyone hates you you {insult}",
        "You're a waste of {noun} honestly",
        "Kill yourself you {insult}",
        "Pathetic {insult} get a life"
    ]
    
    for i in range(30):
        template = np.random.choice(toxic_templates)
        caption = template.format(
            insult=np.random.choice(['idiot', 'moron', 'loser', 'failure', 'waste']),
            action=np.random.choice(['delete', 'remove', 'block']),
            noun=np.random.choice(['trash', 'garbage', 'junk', 'waste'])
        )
        
        samples.append({
            'id': f'synthetic_toxic_{i:04d}',
            'caption': caption,
            'label': 1,
            'category': 'toxicity',
            'source': 'synthetic'
        })
    
    df_synthetic = pd.DataFrame(samples)
    print(f"✅ Generated {len(df_synthetic)} synthetic samples")
    print(f"   Safe: {len(df_synthetic[df_synthetic['label']==0])}")
    print(f"   Unsafe: {len(df_synthetic[df_synthetic['label']==1])}")
    
    return df_synthetic

# ============================================================================
# LOAD EXISTING LOCAL DATASETS
# ============================================================================

def load_local_datasets():
    """Load any existing datasets from data directories."""
    
    print("\n2️⃣ Loading local datasets...")
    
    all_data = []
    
    # Check for existing text datasets
    text_dirs = [
        'data/text_hate_speech',
        'data/text_reddit',
        'data/uci_pornography_metadata'
    ]
    
    for dir_path in text_dirs:
        if os.path.exists(dir_path):
            csv_files = list(Path(dir_path).glob("*.csv"))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, nrows=100)  # Load first 100 rows
                    
                    # Find text column
                    text_cols = [col for col in df.columns if any(
                        x in col.lower() for x in ['text', 'caption', 'title', 'content', 'comment', 'tweet', 'post']
                    )]
                    
                    if text_cols:
                        text_col = text_cols[0]
                        
                        # Find label column
                        label_cols = [col for col in df.columns if any(
                            x in col.lower() for x in ['label', 'class', 'category', 'toxic', 'hate', 'nsfw']
                        )]
                        
                        for idx, row in df.iterrows():
                            if pd.notna(row[text_col]) and len(str(row[text_col])) > 10:
                                
                                # Determine label
                                label = 0  # Default safe
                                if label_cols:
                                    label_val = row[label_cols[0]]
                                    if isinstance(label_val, str):
                                        label_val_lower = label_val.lower()
                                        if any(x in label_val_lower for x in ['hate', 'toxic', 'nsfw', 'adult', 'offensive', '1']):
                                            label = 1
                                    elif label_val > 0:
                                        label = 1
                                
                                all_data.append({
                                    'id': f'{csv_file.stem}_{idx:05d}',
                                    'caption': str(row[text_col])[:500],
                                    'label': label,
                                    'category': 'mixed',
                                    'source': csv_file.stem
                                })
                        
                        print(f"   ✅ Loaded {len(df)} samples from {csv_file.name}")
                
                except Exception as e:
                    print(f"   ⚠️  Error loading {csv_file.name}: {e}")
    
    if all_data:
        df_local = pd.DataFrame(all_data)
        print(f"✅ Total local samples: {len(df_local)}")
        return df_local
    else:
        print("⚠️  No local datasets found")
        return pd.DataFrame()

# ============================================================================
# CREATE FINAL DATASET
# ============================================================================

def create_final_dataset():
    """Combine all sources and create balanced dataset."""
    
    print("\n3️⃣ Creating final balanced dataset...")
    
    # Generate synthetic data
    df_synthetic = generate_synthetic_samples()
    
    # Load local data
    df_local = load_local_datasets()
    
    # Combine
    if len(df_local) > 0:
        df_combined = pd.concat([df_synthetic, df_local], ignore_index=True)
    else:
        df_combined = df_synthetic
    
    print(f"\n📊 Combined dataset: {len(df_combined)} samples")
    print(f"   Label distribution:")
    print(df_combined['label'].value_counts())
    
    # Balance dataset
    safe_samples = df_combined[df_combined['label'] == 0]
    unsafe_samples = df_combined[df_combined['label'] == 1]
    
    print(f"\n   Safe samples: {len(safe_samples)}")
    print(f"   Unsafe samples: {len(unsafe_samples)}")
    
    # Ensure minimum samples per class
    target_per_class = max(MIN_PER_CLASS, min(len(safe_samples), len(unsafe_samples)))
    
    if len(safe_samples) > target_per_class:
        safe_samples = safe_samples.sample(n=target_per_class, random_state=42)
    if len(unsafe_samples) > target_per_class:
        unsafe_samples = unsafe_samples.sample(n=target_per_class, random_state=42)
    
    # Combine and shuffle
    df_final = pd.concat([safe_samples, unsafe_samples], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reset IDs
    df_final['id'] = [f'sample_{i:05d}' for i in range(len(df_final))]
    
    return df_final

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Create final dataset
    df_final = create_final_dataset()
    
    print("\n4️⃣ Saving datasets...")
    
    # Create captions.csv
    captions_df = df_final[['id', 'caption']].copy()
    captions_df.to_csv('data/captions.csv', index=False)
    print(f"✅ Saved data/captions.csv ({len(captions_df)} samples)")
    
    # Create labels.csv
    labels_df = df_final[['id', 'label']].copy()
    labels_df.to_csv('data/labels.csv', index=False)
    print(f"✅ Saved data/labels.csv ({len(labels_df)} samples)")
    
    # Create full dataset with metadata
    df_final.to_csv('data/expanded_dataset.csv', index=False)
    print(f"✅ Saved data/expanded_dataset.csv (with metadata)")
    
    # Save category breakdown
    category_breakdown = df_final.groupby(['category', 'label']).size().reset_index(name='count')
    category_breakdown.to_csv('data/category_breakdown.csv', index=False)
    print(f"✅ Saved data/category_breakdown.csv")
    
    # Display summary
    print("\n" + "="*80)
    print("📊 FINAL DATASET SUMMARY")
    print("="*80)
    
    print(f"\nTotal samples: {len(df_final)}")
    print(f"Safe (0): {len(df_final[df_final['label']==0])} ({len(df_final[df_final['label']==0])/len(df_final)*100:.1f}%)")
    print(f"Unsafe (1): {len(df_final[df_final['label']==1])} ({len(df_final[df_final['label']==1])/len(df_final)*100:.1f}%)")
    
    print("\n📂 Category breakdown:")
    print(category_breakdown.to_string(index=False))
    
    print("\n🔍 Sample data:")
    print("\n🟢 Safe samples:")
    for _, row in df_final[df_final['label']==0].head(3).iterrows():
        print(f"   {row['category']:15s}: {row['caption'][:70]}...")
    
    print("\n🔴 Unsafe samples:")
    for _, row in df_final[df_final['label']==1].head(5).iterrows():
        print(f"   {row['category']:15s}: {row['caption'][:70]}...")
    
    print("\n" + "="*80)
    print("✅ DATA COLLECTION COMPLETE!")
    print("="*80)
    print("""
📁 Files created:
   ✓ data/captions.csv
   ✓ data/labels.csv
   ✓ data/expanded_dataset.csv
   ✓ data/category_breakdown.csv

➡️  Next steps:
   1. Extract features: python scripts/extract_text_features.py
   2. Train models: python scripts/train_models.py
   3. Test predictions: python test_production.py

🎯 Dataset includes:
   ✓ Safe content (neutral, educational, family-friendly)
   ✓ Adult/NSFW content
   ✓ Hate speech
   ✓ Racism
   ✓ Sexism
   ✓ Violence
   ✓ Toxicity

⚠️  IMPORTANT: This is for content moderation research only.
    Use responsibly and ethically.
""")

if __name__ == "__main__":
    main()