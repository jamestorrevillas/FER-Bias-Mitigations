# bias_mitigation_approaches/multi_task_learning/utils/data_loader.py

import numpy as np
import pandas as pd
import os
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import sys
from sklearn.utils import shuffle

# Append parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def custom_normalization(image):
    """
    Match training normalization used in the FER+ model
    
    Args:
        image: Input image to normalize
        
    Returns:
        Normalized image in the range [-1, 1]
    """
    image = image.astype('float32')
    image = image / 255.0
    image = (image - 0.5) * 2.0
    return image

def load_original_rafdb_dataset(split='train'):
    """
    Load original RAF-DB dataset with demographic information
    
    Args:
        split: 'train' or 'test' dataset split to load
    
    Returns:
        Tuple of (images, emotion_labels, gender_labels, age_labels)
    """
    print(f"Loading original RAF-DB {split} dataset...")
    
    # Select appropriate paths based on split
    if split == 'train':
        labels_path = RAFDB_TRAIN_LABELS_PATH
        dataset_path = RAFDB_TRAIN_DIR
    else:
        labels_path = RAFDB_TEST_LABELS_PATH
        dataset_path = RAFDB_TEST_DIR
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found at {dataset_path}")
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    
    images = []
    emotion_labels = []
    gender_labels = []
    age_labels = []
    demographic_info = []
    failed_images = []
    
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Loading RAF-DB {split} images"):
        img_path = os.path.join(dataset_path, str(row['label']), row['image'])
        try:
            img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
            img_array = img_to_array(img)
            img_array = custom_normalization(img_array)  # Apply normalization
            images.append(img_array)
            
            # Extract labels
            emotion_labels.append(row['label'])  # 1-based indexing as in RAF-DB
            gender_labels.append(row['Gender'])
            age_labels.append(row['Age_Group'])
            
            # Store demographic info for later use
            demographic_info.append({
                'gender': row['Gender'],
                'age': row['Age_Group']
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            failed_images.append(img_path)
    
    if failed_images:
        print(f"Failed to load {len(failed_images)} images")
    
    return np.array(images), np.array(emotion_labels), np.array(gender_labels), np.array(age_labels), demographic_info

def load_augmented_rafdb_dataset():
    """
    Load augmented RAF-DB dataset
    
    Returns:
        Tuple of (augmented_images, augmented_emotion_labels, augmented_gender_labels, augmented_age_labels)
    """
    print("Loading augmented RAF-DB dataset...")
    
    if not os.path.exists(RAFDB_AUGMENTED_IMAGES_PATH):
        raise FileNotFoundError(f"Augmented images file not found at {RAFDB_AUGMENTED_IMAGES_PATH}")
    
    if not os.path.exists(RAFDB_AUGMENTED_LABELS_PATH):
        raise FileNotFoundError(f"Augmented labels file not found at {RAFDB_AUGMENTED_LABELS_PATH}")
    
    if not os.path.exists(RAFDB_AUGMENTED_DEMOGRAPHICS_PATH):
        raise FileNotFoundError(f"Augmented demographics file not found at {RAFDB_AUGMENTED_DEMOGRAPHICS_PATH}")
    
    try:
        # Load augmented images
        augmented_images = np.load(RAFDB_AUGMENTED_IMAGES_PATH)
        print(f"Loaded {len(augmented_images)} augmented images with shape {augmented_images.shape}")
        
        # Load augmented labels
        augmented_emotion_labels = np.load(RAFDB_AUGMENTED_LABELS_PATH)
        print(f"Loaded {len(augmented_emotion_labels)} augmented emotion labels")
        
        # Load demographic info
        with open(RAFDB_AUGMENTED_DEMOGRAPHICS_PATH, 'rb') as f:
            augmented_demographics = pickle.load(f)
        print(f"Loaded demographic information for {len(augmented_demographics)} augmented samples")
        
        # Extract gender and age from demographic info
        augmented_gender_labels = np.array([demo['gender'] for demo in augmented_demographics])
        augmented_age_labels = np.array([demo['age'] for demo in augmented_demographics])
        
        return augmented_images, augmented_emotion_labels, augmented_gender_labels, augmented_age_labels, augmented_demographics
        
    except Exception as e:
        print(f"Error loading augmented RAF-DB dataset: {str(e)}")
        return None, None, None, None, None

def balance_dataset_by_demographics(images, emotions, gender, age):
    """
    Balance dataset to have equal representation across demographic groups
    
    Args:
        images: Image data
        emotions: Emotion labels
        gender: Gender labels
        age: Age labels
        
    Returns:
        Balanced dataset (images, emotions, gender, age)
    """
    print("Balancing dataset by demographic attributes...")
    
    # Find counts for each demographic group
    gender_counts = np.bincount(gender)
    print(f"Gender distribution before balancing: {gender_counts}")
    
    # Balance by gender (assuming binary gender 0/1)
    # Find the minority group
    min_gender_count = np.min(gender_counts)
    
    # Extract indices for each gender
    male_indices = np.where(gender == 0)[0]
    female_indices = np.where(gender == 1)[0]
    
    # Randomly sample from the majority group to match minority
    if len(male_indices) > min_gender_count:
        male_indices = np.random.choice(male_indices, min_gender_count, replace=False)
    if len(female_indices) > min_gender_count:
        female_indices = np.random.choice(female_indices, min_gender_count, replace=False)
    
    # Combine balanced indices
    balanced_indices = np.concatenate([male_indices, female_indices])
    
    # Shuffle the indices to avoid order bias
    np.random.shuffle(balanced_indices)
    
    # Extract the balanced dataset
    balanced_images = images[balanced_indices]
    balanced_emotions = emotions[balanced_indices]
    balanced_gender = gender[balanced_indices]
    balanced_age = age[balanced_indices]
    
    print(f"Dataset balanced: {len(balanced_images)} samples")
    print(f"Balanced gender distribution: {np.bincount(balanced_gender)}")
    
    return balanced_images, balanced_emotions, balanced_gender, balanced_age

def load_combined_rafdb_dataset(split='train', balance_demographics=False):
    """
    Load both original and augmented RAF-DB datasets and combine them
    
    Args:
        split: 'train' or 'test' dataset split to load
        balance_demographics: Whether to balance dataset by demographic attributes
        
    Returns:
        Tuple of (combined_images, emotion_labels, gender_labels, age_labels, demographic_info)
    """
    # Load original dataset
    original_images, original_emotions, original_gender, original_age, original_demo_info = load_original_rafdb_dataset(split)
    
    # Only load augmented data for training split
    if split == 'train':
        # Load augmented dataset
        augmented_images, augmented_emotions, augmented_gender, augmented_age, augmented_demo_info = load_augmented_rafdb_dataset()
        
        if augmented_images is not None:
            # Combine datasets
            combined_images = np.concatenate([original_images, augmented_images])
            combined_emotions = np.concatenate([original_emotions, augmented_emotions])
            combined_gender = np.concatenate([original_gender, augmented_gender])
            combined_age = np.concatenate([original_age, augmented_age])
            
            # Combine demographic info lists
            combined_demo_info = original_demo_info.copy()
            combined_demo_info.extend(augmented_demo_info)
            
            print(f"Combined dataset: {len(combined_images)} samples")
            print(f"  Original: {len(original_images)} samples")
            print(f"  Augmented: {len(augmented_images)} samples")
            
            # Apply demographic balancing if requested
            if balance_demographics:
                combined_images, combined_emotions, combined_gender, combined_age = balance_dataset_by_demographics(
                    combined_images, combined_emotions, combined_gender, combined_age
                )
                
                # Update demographic info to match balanced dataset
                # This is an approximation, as we don't have a direct mapping back to the original info
                # In a real implementation, you would want to maintain this mapping
                combined_demo_info = []
                for i in range(len(combined_gender)):
                    combined_demo_info.append({
                        'gender': combined_gender[i],
                        'age': combined_age[i]
                    })
            
            return combined_images, combined_emotions, combined_gender, combined_age, combined_demo_info
        else:
            print("Warning: Could not load augmented data. Using original data only.")
            return original_images, original_emotions, original_gender, original_age, original_demo_info
    else:
        # For test split, return only original data (never augment test data)
        return original_images, original_emotions, original_gender, original_age, original_demo_info

def convert_rafdb_to_ferplus_emotions(rafdb_emotions):
    """
    Convert RAF-DB emotion indices to FER+ emotion indices
    
    Args:
        rafdb_emotions: Array of RAF-DB emotion indices (1-7)
        
    Returns:
        Array of FER+ emotion indices (0-7)
    """
    return np.array([RAFDB_TO_FERPLUS[e] for e in rafdb_emotions])

def demographic_stratified_split(images, emotions, gender, age, val_split=VALIDATION_SPLIT):
    """
    Split dataset while maintaining demographic and emotion distributions
    
    Args:
        images: Input images
        emotions: Emotion labels
        gender: Gender labels
        age: Age labels
        val_split: Validation split ratio
        
    Returns:
        Tuple of (train_images, val_images, train_emotions, val_emotions,
                 train_gender, val_gender, train_age, val_age)
    """
    print("Performing demographic-stratified split...")
    
    # Create a stratification column that combines emotion and demographics
    # We'll use a simple way to combine them
    emotion_gender = emotions * 10 + gender  # This works because gender is binary (0/1)
    
    # Split based on this combined attribute
    indices = np.arange(len(images))
    from sklearn.model_selection import train_test_split
    
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=42,
        stratify=emotion_gender
    )
    
    # Extract data using indices
    train_images = images[train_idx]
    val_images = images[val_idx]
    train_emotions = emotions[train_idx]
    val_emotions = emotions[val_idx]
    train_gender = gender[train_idx]
    val_gender = gender[val_idx]
    train_age = age[train_idx]
    val_age = age[val_idx]
    
    # Print statistics
    print(f"Split dataset: {len(train_images)} training samples, {len(val_images)} validation samples")
    print(f"Training gender distribution: {np.bincount(train_gender)}")
    print(f"Validation gender distribution: {np.bincount(val_gender)}")
    
    return train_images, val_images, train_emotions, val_emotions, train_gender, val_gender, train_age, val_age

def prepare_multi_task_data(images, emotion_labels, gender_labels, age_labels):
    """
    Prepare data for multi-task learning by converting to one-hot encoding
    
    Args:
        images: Input images
        emotion_labels: Emotion labels (in RAF-DB format)
        gender_labels: Gender labels
        age_labels: Age labels
        
    Returns:
        Tuple of (x, [y_emotion, y_gender, y_age])
    """
    import tensorflow as tf
    
    # Convert RAF-DB emotion labels to FER+ format
    ferplus_emotion_labels = convert_rafdb_to_ferplus_emotions(emotion_labels)
    
    # Convert to one-hot encoding
    y_emotion = tf.keras.utils.to_categorical(ferplus_emotion_labels, NUM_EMOTION_CLASSES)
    y_gender = tf.keras.utils.to_categorical(gender_labels, NUM_GENDER_CLASSES)
    
    # Fix for age labels - adjust to be 0-indexed for to_categorical
    # Age classes are 1-indexed in RAF-DB (1-5), but to_categorical expects 0-indexed
    age_labels_adjusted = age_labels - 1  # Adjust from 1-5 to 0-4
    y_age = tf.keras.utils.to_categorical(age_labels_adjusted, NUM_AGE_CLASSES)
    
    return images, [y_emotion, y_gender, y_age]

def generate_balanced_batches(images, emotion_labels, gender_labels, age_labels, batch_size=BATCH_SIZE):
    """
    Generator function to yield balanced batches based on demographics
    
    Args:
        images: Input images
        emotion_labels: Emotion labels
        gender_labels: Gender labels
        age_labels: Age labels
        batch_size: Size of each batch
    
    Yields:
        Tuple of (batch_images, [batch_emotion, batch_gender, batch_age])
    """
    # Convert labels to one-hot encoding
    x, [y_emotion, y_gender, y_age] = prepare_multi_task_data(
        images, emotion_labels, gender_labels, age_labels
    )
    
    # Get indices for each gender
    male_indices = np.where(gender_labels == 0)[0]
    female_indices = np.where(gender_labels == 1)[0]
    
    # Calculate number of samples per gender in each batch
    samples_per_gender = batch_size // 2
    
    # Shuffle indices
    np.random.shuffle(male_indices)
    np.random.shuffle(female_indices)
    
    # Track position in each gender array
    male_pos = 0
    female_pos = 0
    
    while True:
        # Reset positions if we've gone through all samples
        if male_pos + samples_per_gender >= len(male_indices):
            np.random.shuffle(male_indices)
            male_pos = 0
        
        if female_pos + samples_per_gender >= len(female_indices):
            np.random.shuffle(female_indices)
            female_pos = 0
        
        # Get batch indices
        batch_male_indices = male_indices[male_pos:male_pos + samples_per_gender]
        batch_female_indices = female_indices[female_pos:female_pos + samples_per_gender]
        batch_indices = np.concatenate([batch_male_indices, batch_female_indices])
        
        # Shuffle to avoid order bias
        np.random.shuffle(batch_indices)
        
        # Create batch
        batch_x = x[batch_indices]
        batch_emotion = y_emotion[batch_indices]
        batch_gender = y_gender[batch_indices]
        batch_age = y_age[batch_indices]
        
        # Update positions
        male_pos += samples_per_gender
        female_pos += samples_per_gender
        
        yield batch_x, [batch_emotion, batch_gender, batch_age]