# utils/dataset_augmentation/rafdb_augment_dataset.py

"""
This script generates augmented images from the RAF-DB training dataset
using a targeted approach that prioritizes underrepresented demographic groups
and emotion classes. Now using the automatic augmentation framework.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from our new modules
from utils.dataset_augmentation import distribution_analyzer
from utils.dataset_augmentation import image_processor
from utils.dataset_augmentation import config_settings

# Define paths (keep these the same as before)
DATASET_DIR = 'resources/dataset/raf-db'
LABELS_PATH = os.path.join(DATASET_DIR, 'labels', 'train_labels.csv')
DATASET_PATH = os.path.join(DATASET_DIR, 'dataset', 'train')
OUTPUT_DIR = os.path.join(DATASET_DIR, 'augmented')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'augmented_images.npy')
STATS_DIR = os.path.join(OUTPUT_DIR, 'augmentation_stats')

# Create required directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)
os.makedirs(os.path.join(STATS_DIR, 'distributions'), exist_ok=True)

def load_dataset_with_demographics(labels_path, dataset_path):
    """Load original images, labels, and demographic information"""
    print("Loading original images and demographic information...")
    labels_df = pd.read_csv(labels_path)
    
    images = []
    labels = []
    demographic_info = []
    failed_images = []
    
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Loading images"):
        img_path = os.path.join(dataset_path, str(row['label']), row['image'])
        try:
            img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(row['label'])  # 1-based indexing as in RAF-DB
            
            # Get demographic information
            demo_info = {
                'gender': row['Gender'],
                'age': row['Age_Group']
            }
            demographic_info.append(demo_info)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            failed_images.append(img_path)
    
    if failed_images:
        print(f"Failed to load {len(failed_images)} images")
    
    return np.array(images), np.array(labels), demographic_info

def main():
    # Load dataset with demographic information
    images, labels, demographic_info = load_dataset_with_demographics(LABELS_PATH, DATASET_PATH)
    print(f"Loaded {len(images)} original images")
    
    # ----------------- NEW CODE USING OUR FRAMEWORK -----------------
    
    # Analyze dataset distributions
    print("Analyzing dataset distributions...")
    emotion_distribution = distribution_analyzer.analyze_emotion_distribution(labels)
    demographic_distribution = distribution_analyzer.analyze_demographic_distribution(demographic_info)
    intersection_distribution = distribution_analyzer.analyze_intersectional_distribution(
        labels, demographic_info
    )
    
    # Create distribution visualizations
    image_processor.plot_distribution(
        emotion_distribution, 
        'Emotion Distribution', 
        os.path.join(STATS_DIR, 'distributions', 'emotion_distribution.png'),
        [config_settings.RAFDB_EMOTIONS.get(i, i) for i in range(1, 8)]
    )
    
    # Generate comprehensive augmentation plan
    print("Generating intersectional augmentation plan...")
    augmentation_plan = distribution_analyzer.generate_intersectional_plan(
        emotion_distribution,
        demographic_distribution,
        intersection_distribution
    )
    
    # Print the plan in human-readable format
    distribution_analyzer.print_augmentation_plan(augmentation_plan['base_emotion_plan'])
    
    # Generate augmentations
    print("Generating augmentations...")
    augmented_images, augmented_labels, original_indices, augmented_demographics = \
        image_processor.generate_augmentations(
            images, labels, augmentation_plan, demographic_info
        )
    print(f"Generated {len(augmented_images)} augmented images")
    
    # Track augmentation counts
    emotion_counts = {label: 0 for label in np.unique(augmented_labels)}
    gender_counts = {0: 0, 1: 0}  # Male: 0, Female: 1
    age_counts = {age: 0 for age in range(1, 6)}  # Age groups 1-5
    
    for i, label in enumerate(augmented_labels):
        # Track emotion counts
        emotion_counts[label] += 1
        
        # Track demographic counts if available
        if i < len(augmented_demographics):
            demo = augmented_demographics[i]
            gender = demo['gender']
            age = demo['age']
            
            if gender in gender_counts:
                gender_counts[gender] += 1
            
            if age in age_counts:
                age_counts[age] += 1
    
    # Create augmentation statistics plots
    image_processor.create_augmentation_stats_plot(
        emotion_counts,
        config_settings.RAFDB_EMOTIONS,
        os.path.join(STATS_DIR, 'emotion_augmentation_stats.png')
    )
    
    # Visualize results
    image_processor.plot_augmentation_plan(
        emotion_distribution,
        augmentation_plan['base_emotion_plan'],
        os.path.join(STATS_DIR, 'count_comparison.png')
    )
    
    # Visualize some sample augmentations
    if len(augmented_images) > 0:
        sample_indices = np.random.choice(
            len(original_indices), 
            min(10, len(original_indices)), 
            replace=False
        )
        image_processor.visualize_samples(
            images, 
            augmented_images, 
            [original_indices[i] for i in sample_indices],
            os.path.join(STATS_DIR, 'augmentation_samples.png')
        )
    
    # Apply normalization to match training pipeline
    normalized_augmented_images = image_processor.apply_custom_normalization(augmented_images)
    
    # Save augmented images and metadata
    np.save(OUTPUT_PATH, normalized_augmented_images)
    np.save(os.path.join(OUTPUT_DIR, 'augmented_labels.npy'), augmented_labels)
    np.save(os.path.join(OUTPUT_DIR, 'original_indices.npy'), np.array(original_indices))
    
    # Save demographic info using pickle
    with open(os.path.join(OUTPUT_DIR, 'augmented_demographics.pkl'), 'wb') as f:
        pickle.dump(augmented_demographics, f)
    
    # Save augmentation plan for reference
    with open(os.path.join(OUTPUT_DIR, 'augmentation_plan.pkl'), 'wb') as f:
        pickle.dump(augmentation_plan, f)
    
    print(f"Augmented images saved to {OUTPUT_PATH}")
    print(f"Labels saved to {os.path.join(OUTPUT_DIR, 'augmented_labels.npy')}")
    print(f"Original indices saved to {os.path.join(OUTPUT_DIR, 'original_indices.npy')}")
    print(f"Demographic info saved to {os.path.join(OUTPUT_DIR, 'augmented_demographics.pkl')}")
    
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