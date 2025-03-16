# bias_mitigation_approaches/dynamic_weighting/utils/data_loader.py

import numpy as np
import pandas as pd
import os
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from . import config

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

def load_augmented_fer_dataset():
    """
    Load augmented FER+ dataset from numpy files
    
    Returns:
        Tuple of (augmented_images, augmented_labels)
    """
    print(f"Loading augmented FER+ dataset...")
    
    if not os.path.exists(config.FER_AUGMENTED_IMAGES_PATH):
        raise FileNotFoundError(f"Augmented images file not found at {config.FER_AUGMENTED_IMAGES_PATH}")
    
    if not os.path.exists(config.FER_AUGMENTED_LABELS_PATH):
        raise FileNotFoundError(f"Augmented labels file not found at {config.FER_AUGMENTED_LABELS_PATH}")
    
    try:
        # Load augmented images
        augmented_images = np.load(config.FER_AUGMENTED_IMAGES_PATH)
        print(f"Loaded {len(augmented_images)} augmented images with shape {augmented_images.shape}")
        
        # Load augmented labels
        augmented_labels = np.load(config.FER_AUGMENTED_LABELS_PATH)
        print(f"Loaded {len(augmented_labels)} augmented labels")
        
        if len(augmented_images) != len(augmented_labels):
            print(f"Warning: Mismatch between number of images and labels!")
        
        return augmented_images, augmented_labels
        
    except Exception as e:
        print(f"Error loading augmented dataset: {str(e)}")
        return None, None

def process_ferplus_data(dataset_df):
    """
    Process FER+ images and labels from DataFrame
    
    Args:
        dataset_df: DataFrame containing FER+ dataset
        
    Returns:
        Tuple of (normalized_images, labels)
    """
    # Extract images and labels
    images = []
    labels = []

    for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Processing images"):
        # Process image data
        pixels = [int(p) for p in row['image'].split()]
        img = np.array(pixels, dtype=np.uint8).reshape(48, 48, 1)
        images.append(img)
        
        # Get emotion votes and determine majority
        votes = row[config.EMOTION_COLUMNS].values
        majority_idx = np.argmax(votes)
        labels.append(majority_idx)
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Processed {len(images)} images and {len(labels)} labels")
    
    # Apply normalization
    normalized_images = np.array([custom_normalization(img) for img in tqdm(images, desc="Normalizing")])
    
    return normalized_images, labels

def load_ferplus_dataset():
    """
    Load FER+ dataset from CSV and split into train/test sets
    
    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels)
    """
    print(f"Loading FER+ dataset from {config.DATASET_CSV_PATH}...")
    
    if not os.path.exists(config.DATASET_CSV_PATH):
        raise FileNotFoundError(f"Dataset CSV file not found at {config.DATASET_CSV_PATH}")
    
    try:
        # Load dataset CSV
        dataset_df = pd.read_csv(config.DATASET_CSV_PATH)
        print(f"Loaded dataset with {len(dataset_df)} entries")
        
        # Split into train and test sets
        train_df = dataset_df[dataset_df['dataset'] == 'train']
        test_df = dataset_df[dataset_df['dataset'] == 'test']
        
        print(f"Training set has {len(train_df)} entries")
        print(f"Test set has {len(test_df)} entries")
        
        # Process train and test data
        print("Processing training data...")
        train_images, train_labels = process_ferplus_data(train_df)
        
        print("Processing test data...")
        test_images, test_labels = process_ferplus_data(test_df)
        
        return train_images, train_labels, test_images, test_labels
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None, None, None

def load_rafdb_dataset(split='train'):
    """
    Load RAF-DB dataset with demographic information
    
    Args:
        split: 'train' or 'test' dataset split to load
    
    Returns:
        Tuple of (images, labels, demographic_info)
    """
    print(f"Loading RAF-DB {split} dataset...")
    
    # Select appropriate paths based on split
    if split == 'train':
        labels_path = config.RAFDB_TRAIN_LABELS_PATH
        dataset_path = config.RAFDB_TRAIN_DIR
    else:
        labels_path = config.RAFDB_TEST_LABELS_PATH
        dataset_path = config.RAFDB_TEST_DIR
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found at {dataset_path}")
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    
    images = []
    labels = []
    demographic_info = []
    failed_images = []
    
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Loading RAF-DB {split} images"):
        img_path = os.path.join(dataset_path, str(row['label']), row['image'])
        try:
            img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
            img_array = img_to_array(img)
            img_array = custom_normalization(img_array)  # Apply normalization
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

def prepare_combined_data(original_images, original_labels, augmented_images, augmented_labels):
    """
    Prepare training data by combining original and augmented data
    
    Args:
        original_images: Original training images
        original_labels: Original training labels
        augmented_images: Augmented images to add to training set
        augmented_labels: Augmented labels to add to training set
        
    Returns:
        Tuple of (combined_images, combined_labels)
    """
    # Make sure we have data to combine
    if original_images is None or augmented_images is None:
        print("Warning: Missing data. Using only available data.")
        if original_images is not None:
            return original_images, original_labels
        elif augmented_images is not None:
            return augmented_images, augmented_labels
        else:
            raise ValueError("No data available for training.")
    
    # Combine original and augmented data
    X = np.concatenate([original_images, augmented_images], axis=0)
    y = np.concatenate([original_labels, augmented_labels], axis=0)
    
    print(f"Combined dataset shape: {X.shape}, Labels shape: {y.shape}")
    
    # Print emotion distribution in the combined dataset
    print("\nEmotion distribution in combined dataset:")
    for i in range(config.NUM_CLASSES):
        count = np.sum(y == i)
        percentage = (count / len(y)) * 100
        emotion_name = config.FERPLUS_EMOTIONS.get(i, f"Emotion {i}")
        print(f"  {emotion_name}: {count} samples ({percentage:.1f}%)")
    
    return X, y

def create_weighted_dataset(images, labels, sample_weights=None):
    """
    Create a weighted TensorFlow dataset for training
    
    Args:
        images: Training images
        labels: Training labels
        sample_weights: Optional sample weights (default: uniform weights)
        
    Returns:
        TensorFlow dataset with sample weights
    """
    import tensorflow as tf
    
    # Create one-hot encoded labels
    one_hot_labels = tf.keras.utils.to_categorical(labels, config.NUM_CLASSES)
    
    # Create weights if not provided
    if sample_weights is None:
        sample_weights = np.ones(len(labels))
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        images, one_hot_labels, sample_weights
    ))
    
    return dataset