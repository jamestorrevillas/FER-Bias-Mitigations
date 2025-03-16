# utils/dataset_augmentation/ferplus_augment_dataset.py

"""
This script generates augmented images from the FER+ dataset
using a distribution-aware approach that balances emotion classes.
Now using the automatic augmentation framework.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from our new modules
from utils.dataset_augmentation import distribution_analyzer
from utils.dataset_augmentation import image_processor
from utils.dataset_augmentation import config_settings

# Define paths (keep these the same as before)
DATASET_DIR = 'resources/dataset/fer'
FER_CSV_PATH = os.path.join(DATASET_DIR, 'labels', 'fer2013.csv')
FERPLUS_CSV_PATH = os.path.join(DATASET_DIR, 'labels', 'fer2013new.csv')
DATASET_CSV_PATH = os.path.join(DATASET_DIR, 'labels', 'dataset.csv')
OUTPUT_DIR = os.path.join(DATASET_DIR, 'augmented')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'augmented_images.npy')
STATS_DIR = os.path.join(OUTPUT_DIR, 'augmentation_stats')

# Create required directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

# Define emotion mappings
EMOTION_LABELS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
COLUMN_NAMES = ['dataset', 'image', 'fer_code'] + EMOTION_LABELS + ['unknown', 'no-face']

def load_ferplus_dataset():
    """Load and process the FER+ dataset"""
    print("Loading FER+ dataset...")
    
    # Check if processed dataset exists
    if os.path.exists(DATASET_CSV_PATH):
        print("Loading preprocessed dataset...")
        dataset_df = pd.read_csv(DATASET_CSV_PATH)
    else:
        print("Processing raw FER+ files...")
        # Create the unified dataset from FER and FER+ files
        fer_df = pd.read_csv(FER_CSV_PATH)
        ferplus_df = pd.read_csv(FERPLUS_CSV_PATH)
        
        # Process the files to create dataset.csv (simplified version)
        dataset_df = pd.DataFrame(columns=COLUMN_NAMES)
        
        # In actual implementation, this would use the proper preprocessing
        # from the existing code in data.py and dataset.py
        print("ERROR: dataset.csv not found. Please run the dataset preprocessing first.")
        return None
    
    # Filter to only include training data
    train_df = dataset_df[dataset_df['dataset'] == 'train']
    
    print(f"Loaded {len(train_df)} training samples")
    return train_df

def process_images_and_labels(dataset_df):
    """Extract images and labels from the dataset"""
    IMG_SIZE = 48
    
    # Extract emotion votes and determine majority emotion
    emotion_columns = EMOTION_LABELS
    
    # Convert string pixel values to numpy arrays
    images = []
    majority_labels = []
    emotion_votes = []
    
    print("Processing images and labels...")
    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
        # Process image data
        pixels = [int(p) for p in row['image'].split()]
        img = np.array(pixels, dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE, 1)
        images.append(img)
        
        # Get emotion votes and determine majority
        votes = row[emotion_columns].values
        emotion_votes.append(votes)
        majority_idx = np.argmax(votes)
        majority_labels.append(majority_idx)
    
    return np.array(images), np.array(majority_labels), np.array(emotion_votes)

def main():
    # Load FER+ dataset
    dataset_df = load_ferplus_dataset()
    if dataset_df is None:
        print("ERROR: Could not load dataset. Please ensure dataset.csv is available.")
        return
    
    # Process images and labels
    images, labels, emotion_votes = process_images_and_labels(dataset_df)
    print(f"Processed {len(images)} images with labels")
    
    # ----------------- NEW CODE USING OUR FRAMEWORK -----------------
    
    # Analyze dataset distribution automatically
    print("Analyzing dataset distribution...")
    emotion_distribution = distribution_analyzer.analyze_emotion_distribution(labels)
    
    # Generate augmentation plan automatically
    print("Generating automatic augmentation plan...")
    augmentation_plan = distribution_analyzer.generate_emotion_plan(emotion_distribution)
    
    # Print the plan in human-readable format
    distribution_analyzer.print_augmentation_plan(augmentation_plan)
    
    # Generate augmentations
    print("Generating augmentations...")
    augmented_images, augmented_labels, original_indices, _ = image_processor.generate_augmentations(
        images, labels, augmentation_plan, emotion_votes
    )
    print(f"Generated {len(augmented_images)} augmented images")
    
    # Visualize distributions
    image_processor.plot_augmentation_plan(
        emotion_distribution,
        augmentation_plan,
        os.path.join(STATS_DIR, 'count_comparison.png')
    )
    
    # Visualize some sample augmentations
    if len(augmented_images) > 0:
        sample_indices = np.random.choice(len(original_indices), min(10, len(original_indices)), replace=False)
        image_processor.visualize_samples(
            images, 
            augmented_images, 
            [original_indices[i] for i in sample_indices],
            os.path.join(STATS_DIR, 'augmentation_samples.png')
        )
    
    # Track augmentation counts by emotion
    augmentation_counts = {label: 0 for label in np.unique(augmented_labels)}
    for label in augmented_labels:
        augmentation_counts[label] += 1
    
    # Create augmentation statistics plot
    image_processor.create_augmentation_stats_plot(
        augmentation_counts,
        config_settings.FERPLUS_EMOTIONS,
        os.path.join(STATS_DIR, 'emotion_augmentation_stats.png')
    )
    
    # Apply normalization to match training pipeline
    normalized_augmented_images = image_processor.apply_custom_normalization(augmented_images)
    
    # Save augmented images and metadata
    np.save(OUTPUT_PATH, normalized_augmented_images)
    np.save(os.path.join(OUTPUT_DIR, 'augmented_labels.npy'), augmented_labels)
    np.save(os.path.join(OUTPUT_DIR, 'augmented_votes.npy'), emotion_votes)
    np.save(os.path.join(OUTPUT_DIR, 'original_indices.npy'), np.array(original_indices))
    
    # Save augmentation plan for reference
    with open(os.path.join(OUTPUT_DIR, 'augmentation_plan.pkl'), 'wb') as f:
        pickle.dump(augmentation_plan, f)
    
    print(f"Augmented images saved to {OUTPUT_PATH}")
    print(f"Labels saved to {os.path.join(OUTPUT_DIR, 'augmented_labels.npy')}")
    print(f"Votes saved to {os.path.join(OUTPUT_DIR, 'augmented_votes.npy')}")
    print(f"Original indices saved to {os.path.join(OUTPUT_DIR, 'original_indices.npy')}")
    
    # Report memory usage and statistics
    original_size_mb = images.nbytes / (1024 * 1024)
    augmented_size_mb = normalized_augmented_images.nbytes / (1024 * 1024)
    
    print("\nAugmentation Summary:")
    print(f"Original images: {len(images):,} ({original_size_mb:.2f} MB)")
    print(f"Augmented images: {len(augmented_images):,} ({augmented_size_mb:.2f} MB)")
    print(f"Augmentation factor: {len(augmented_images)/len(images):.2f}x")
    print(f"Statistics and visualizations saved to {STATS_DIR}")

if __name__ == "__main__":
    main()