# baseline_ferplus_model_training/data/dataset.py

import csv
from itertools import islice
import os
import pandas as pd
import numpy as np

UNIFIED_DATASET_FILE_NAME = 'dataset.csv'

DATASET_NAMES = {'Training'   : 'train',
                 'PublicTest' : 'valid',
                 'PrivateTest': 'test'}

COLUMN_NAMES = ['dataset', 'image', 'fer_code', 'neutral', 'happiness', \
'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', \
'no-face']


def get_dataset_dict(dataset_dir = '../dataset',
                     fer_file_name = 'fer2013.csv',
                     fer_plus_file_name = 'fer2013new.csv',
                     use_augmented_data = False):
    '''Reads the output data csv (creates it first if it doesn't exist) into a
    dict.

    Args:
        dataset_dir(string): a path to a directory with dataset files
        fer_file_name(string): a name of fer csv file
        fer_plus_file_name(string): a name of fer plus csv file
        use_augmented_data(boolean): whether to use the augmented dataset

    Returns: a dictionary of three dataset dataframes ('train', 'valid', 'test').
    '''
    # Check if labels directory exists in dataset_dir
    labels_dir = os.path.join(dataset_dir, 'labels')
    if os.path.exists(labels_dir):
        # If it does, use it for CSV files
        fer_path = os.path.join(labels_dir, fer_file_name)
        ferplus_path = os.path.join(labels_dir, fer_plus_file_name)
        dataset_path = os.path.join(labels_dir, UNIFIED_DATASET_FILE_NAME)
    else:
        # Otherwise, use the main directory
        fer_path = os.path.join(dataset_dir, fer_file_name)
        ferplus_path = os.path.join(dataset_dir, fer_plus_file_name)
        dataset_path = os.path.join(dataset_dir, UNIFIED_DATASET_FILE_NAME)
    
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    
    # Check if the output csv dataset exists
    if os.path.isfile(dataset_path):
        print(f"Loading existing dataset from {dataset_path}")
        dataset_df = pd.read_csv(dataset_path)
    else:
        # Check if the input files exist
        if not os.path.isfile(fer_path):
            raise FileNotFoundError(f"FER file not found: {fer_path}")
        if not os.path.isfile(ferplus_path):
            raise FileNotFoundError(f"FER+ file not found: {ferplus_path}")
            
        print(f"Generating dataset from {fer_path} and {ferplus_path}")
        dataset_df = _generate_dataset_csv(fer_path, ferplus_path, dataset_path)

    dataset_dict = {
        'train': dataset_df.loc[dataset_df['dataset'] == 'train'],
        'valid': dataset_df.loc[dataset_df['dataset'] == 'valid'],
        'test': dataset_df.loc[dataset_df['dataset'] == 'test']
    }
    
    # Augment the dataset if requested
    if use_augmented_data:
        augmented_dir = os.path.join(dataset_dir, 'augmented')
        if os.path.exists(augmented_dir):
            print(f"Looking for augmented data in {augmented_dir}")
            dataset_dict = _add_augmented_data(dataset_dict, augmented_dir)
        else:
            print(f"Warning: Augmented directory not found: {augmented_dir}")
    
    return dataset_dict

def read_dataset_csv(dataset_dir = './'):
    '''Reads into a dataframe a previously generated output dataset csv file.

    Args:
        dataset_dir(string): a path to a directory with dataset files

    Returns: a dataframe containing output dataset.
    '''
    labels_dir = os.path.join(dataset_dir, 'labels')
    if os.path.exists(labels_dir):
        dataset_path = os.path.join(labels_dir, UNIFIED_DATASET_FILE_NAME)
    else:
        dataset_path = os.path.join(dataset_dir, UNIFIED_DATASET_FILE_NAME)
    
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    return pd.read_csv(dataset_path)

def _generate_dataset_csv(fer_path, ferplus_path, dataset_path):
    '''Generates output dataset csv file out of original fer and fer plus files.
    Saves it in the dataset directory.

    Args:
        fer_path(string): full path to fer csv file
        ferplus_path(string): full path to fer plus csv file
        dataset_path(string): full path to output dataset csv file

    Returns: a dataframe containing output dataset.
    '''
    # Create writer
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    output_file = open(dataset_path, 'w')
    writer = csv.DictWriter(output_file, fieldnames=COLUMN_NAMES)
    writer.writeheader()

    # Read ferplus csv
    ferplus_entries = []
    with open(ferplus_path, 'r') as csvfile:
        ferplus_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(ferplus_rows, 1, None):
            ferplus_entries.append(row)

    # While reading fer csv, write to the output dataset csv,
    # combining old data with new labels
    index = 0
    with open(fer_path, 'r') as csvfile:
        fer_rows = csv.reader(csvfile, delimiter=',')
        for row in islice(fer_rows, 1, None):
            ferplus_row = ferplus_entries[index]
            file_name = ferplus_row[1].strip()
            if len(file_name) > 0:
                # dataset type, image string, counts for each emotion
                new_row = {
                    'dataset': DATASET_NAMES[row[2]],
                    'image': str(row[1]),
                    'fer_code': str(row[0]),
                    'neutral': int(ferplus_row[2]),
                    'happiness': int(ferplus_row[3]),
                    'surprise': int(ferplus_row[4]),
                    'sadness': int(ferplus_row[5]),
                    'anger': int(ferplus_row[6]),
                    'disgust': int(ferplus_row[7]),
                    'fear': int(ferplus_row[8]),
                    'contempt': int(ferplus_row[9]),
                    'unknown': int(ferplus_row[10]),
                    'no-face': int(ferplus_row[11])
                }
                writer.writerow(new_row)
            index += 1

    output_file.close()

    # Return dataframe out of created dataset
    return pd.read_csv(dataset_path)

def _add_augmented_data(dataset_dict, augmented_dir):
    '''Adds augmented data to the training set.

    Args:
        dataset_dict(dict): Dictionary containing train/valid/test dataframes
        augmented_dir(string): Path to directory containing augmented data

    Returns:
        Updated dataset dictionary with augmented data added to training set
    '''
    # Check if augmented files exist
    images_path = os.path.join(augmented_dir, 'augmented_images.npy')
    labels_path = os.path.join(augmented_dir, 'augmented_labels.npy')
    votes_path = os.path.join(augmented_dir, 'augmented_votes.npy')
    
    if not os.path.exists(images_path):
        print(f"Warning: Augmented images file not found: {images_path}")
        return dataset_dict
        
    if not os.path.exists(labels_path):
        print(f"Warning: Augmented labels file not found: {labels_path}")
        return dataset_dict
    
    try:
        # Load augmented data - with allow_pickle=True to handle object arrays
        print(f"Loading augmented data from {augmented_dir}")
        aug_images = np.load(images_path, allow_pickle=True)
        aug_labels = np.load(labels_path, allow_pickle=True)
        
        # If votes file exists, load it; otherwise, generate dummy votes
        if os.path.exists(votes_path):
            print(f"Loading augmented votes from {votes_path}")
            aug_votes = np.load(votes_path, allow_pickle=True)  # Fixed to use allow_pickle=True
        else:
            print("Generating dummy vote distributions for augmented data")
            # Create one-hot-style vote distributions
            aug_votes = np.zeros((len(aug_labels), 10))  # 10 emotions including unknown and no-face
            for i, label in enumerate(aug_labels):
                if 0 <= label - 1 < 8:  # Valid emotion label (1-8)
                    aug_votes[i, label - 1] = 10  # 10 votes for the labeled emotion
        
        # Create DataFrame for augmented data
        aug_df = pd.DataFrame(columns=COLUMN_NAMES)
        aug_df['dataset'] = 'train'  # Add to training set
        
        # Process images to match expected format
        def normalize_and_convert(img):
            # Check if normalized [-1,1] or [0,1]
            if img.min() < 0 or img.max() <= 1.0:
                # Convert to [0,255] range
                img = np.clip(((img + 1) / 2 * 255), 0, 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                # Ensure uint8 type
                img = np.clip(img, 0, 255).astype(np.uint8)
            
            # Flatten and convert to string
            return ' '.join(map(str, img.reshape(-1)))
        
        print(f"Processing {len(aug_images)} augmented images...")
        # Process in batches to avoid memory issues
        batch_size = 100
        all_image_strings = []
        
        for i in range(0, len(aug_images), batch_size):
            batch = aug_images[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{len(aug_images)//batch_size + 1}")
            image_strings = [normalize_and_convert(img) for img in batch]
            all_image_strings.extend(image_strings)
        
        aug_df['image'] = all_image_strings
        
        # Add emotion labels as fer_code
        aug_df['fer_code'] = aug_labels
        
        # Add vote distributions
        emotion_cols = COLUMN_NAMES[3:13]  # emotion columns: neutral through no-face
        
        # Ensure votes has the right number of columns
        if aug_votes.shape[1] < len(emotion_cols):
            # Pad with zeros if needed
            padded_votes = np.zeros((aug_votes.shape[0], len(emotion_cols)))
            padded_votes[:, :aug_votes.shape[1]] = aug_votes
            aug_votes = padded_votes
        elif aug_votes.shape[1] > len(emotion_cols):
            # Truncate if needed
            aug_votes = aug_votes[:, :len(emotion_cols)]
        
        # Convert probabilities to counts if needed
        if aug_votes.max() <= 1.0 and aug_votes.min() >= 0:
            aug_votes = (aug_votes * 10).astype(int)
        
        # Add vote columns to DataFrame
        for i, col_name in enumerate(emotion_cols):
            aug_df[col_name] = aug_votes[:, i]
        
        # Concatenate with original training data
        print(f"Adding {len(aug_df)} augmented samples to the training set")
        dataset_dict['train'] = pd.concat([dataset_dict['train'], aug_df], ignore_index=True)
        
        print(f"Final training set size: {len(dataset_dict['train'])} samples")
        
    except Exception as e:
        import traceback
        print(f"Error loading augmented data: {e}")
        traceback.print_exc()
        print("Using original dataset without augmentation.")
    
    return dataset_dict