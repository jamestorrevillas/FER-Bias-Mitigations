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

def create_efficient_dataset(images, labels, weights=None, batch_size=64, shuffle_buffer=10000):
    """
    Create a memory-efficient TensorFlow dataset with prefetching and caching
    
    Args:
        images: Image data
        labels: Labels
        weights: Optional sample weights
        batch_size: Batch size for training
        shuffle_buffer: Buffer size for shuffling
        
    Returns:
        TensorFlow dataset ready for training
    """
    import tensorflow as tf
    
    # Convert labels to one-hot encoding
    one_hot_labels = tf.keras.utils.to_categorical(labels, config.NUM_CLASSES)
    
    # Create dataset
    if weights is not None:
        dataset = tf.data.Dataset.from_tensor_slices((images, one_hot_labels, weights))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((images, one_hot_labels))
    
    # Apply performance optimizations
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=min(len(images), shuffle_buffer))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def extract_fer_demographic_proxy(images, labels, demographic_info_source=None):
    """
    Create proxy demographic information for FER+ dataset based on patterns
    observed in RAF-DB (for intersectional weight adjustments)
    
    Args:
        images: FER+ images
        labels: FER+ labels
        demographic_info_source: Optional source demographic info from RAF-DB
        
    Returns:
        List of demographic info dictionaries
    """
    # Create initial demographic proxies (estimated values)
    demographic_info = []
    
    # Use fixed distributions based on typical patterns
    gender_priors = {
        0: 0.5,  # neutral: 50% male
        1: 0.4,  # happiness: 40% male
        2: 0.5,  # surprise: 50% male
        3: 0.45, # sadness: 45% male
        4: 0.6,  # anger: 60% male
        5: 0.5,  # disgust: 50% male
        6: 0.45, # fear: 45% male
        7: 0.55  # contempt: 55% male
    }
    
    age_priors = {
        0: [0.05, 0.15, 0.5, 0.25, 0.05],  # neutral
        1: [0.1, 0.2, 0.45, 0.2, 0.05],    # happiness
        2: [0.05, 0.2, 0.5, 0.2, 0.05],    # surprise
        3: [0.05, 0.15, 0.45, 0.3, 0.05],  # sadness
        4: [0.05, 0.1, 0.5, 0.3, 0.05],    # anger
        5: [0.05, 0.1, 0.5, 0.3, 0.05],    # disgust
        6: [0.05, 0.2, 0.5, 0.2, 0.05],    # fear
        7: [0.05, 0.1, 0.5, 0.3, 0.05]     # contempt
    }
    
    # If we have demographic source data, compute more accurate estimates
    if demographic_info_source is not None:
        try:
            # Extract demographic information from source
            source_labels = []
            source_genders = []
            source_ages = []
            
            # First collect RAF-DB labels and demographics
            for info in demographic_info_source:
                if 'gender' in info and 'age' in info:
                    source_genders.append(info['gender'])
                    source_ages.append(info['age'])
            
            # Since source_labels needs demographic info and RAF-DB labels, 
            # we need to make sure they're properly aligned
            computed_priors = False
            
            # Skip this refinement if we don't have enough data
            if len(source_genders) > 100:
                # Compute gender priors per emotion
                gender_counts = {}
                age_counts = {}
                
                # Count demographic info by RAF-DB emotion categories
                for i, info in enumerate(demographic_info_source):
                    if i >= len(rafdb_labels):
                        break
                        
                    raf_label = rafdb_labels[i]
                    
                    # Initialize counters if needed
                    if raf_label not in gender_counts:
                        gender_counts[raf_label] = [0, 0]  # [male_count, female_count]
                    
                    if raf_label not in age_counts:
                        age_counts[raf_label] = [0, 0, 0, 0, 0]  # 5 age groups
                    
                    # Add to counters
                    if 'gender' in info:
                        gender = info['gender']
                        if gender in [0, 1]:
                            gender_counts[raf_label][gender] += 1
                    
                    if 'age' in info:
                        age = info['age']
                        if 1 <= age <= 5:
                            age_counts[raf_label][age-1] += 1
                
                # Convert to probabilities and map to FER+ labels
                for raf_label, counts in gender_counts.items():
                    total = sum(counts)
                    if total > 0:
                        male_prob = counts[0] / total
                        
                        # Map to corresponding FER+ emotion index
                        for fer_idx, mapped_raf_idx in config.FERPLUS_TO_RAFDB.items():
                            if mapped_raf_idx == raf_label:
                                gender_priors[fer_idx] = male_prob
                                break
                
                for raf_label, counts in age_counts.items():
                    total = sum(counts)
                    if total > 0:
                        age_prob = [count/total for count in counts]
                        
                        # Map to corresponding FER+ emotion index
                        for fer_idx, mapped_raf_idx in config.FERPLUS_TO_RAFDB.items():
                            if mapped_raf_idx == raf_label:
                                age_priors[fer_idx] = age_prob
                                break
                
                computed_priors = True
            
            if computed_priors:
                print("Using computed demographic priors from RAF-DB")
            else:
                print("Using default demographic priors (not enough RAF-DB data)")
                
        except Exception as e:
            print(f"Warning: Error computing demographic priors: {str(e)}")
            print("Using default demographic priors")
    
    # Generate demographic proxies using priors
    np.random.seed(42)  # For reproducibility
    
    for label in labels:
        # Determine gender (0: male, 1: female)
        male_prob = gender_priors.get(label, 0.5)
        gender = 0 if np.random.random() < male_prob else 1
        
        # Determine age (1-5)
        age_prob = age_priors.get(label, [0.1, 0.2, 0.4, 0.2, 0.1])
        age = np.random.choice([1, 2, 3, 4, 5], p=age_prob)
        
        demographic_info.append({
            'gender': gender,
            'age': age
        })
    
    return demographic_info