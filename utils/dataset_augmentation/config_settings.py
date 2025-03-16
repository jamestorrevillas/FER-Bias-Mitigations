# utils/dataset_augmentation/config_settings.py

"""Configuration parameters for automatic data augmentation."""

# Target balance parameters
TARGET_IMBALANCE_RATIO = 3.5  # Aim to reduce max:min ratio to this value
MAX_AUGMENTATION_FACTOR = 25.0  # Maximum multiplication factor for any class

# Quality-aware parameters
HIGH_AGREEMENT_THRESHOLD = 0.67  # Threshold for FER+ voting agreement
HIGH_AGREEMENT_BOOST = 1.5  # Multiplier for high agreement samples
VERY_HIGH_AGREEMENT_THRESHOLD = 0.85  # Threshold for extremely clear expressions 
VERY_HIGH_AGREEMENT_BOOST = 2.0  # Higher multiplier for very clear expressions

# Intersection priority settings
CRITICAL_INTERSECTION_THRESHOLD = 30  # Intersections with fewer samples are "critical"
CRITICAL_INTERSECTION_BOOST = 1.5  # Additional boost for critical intersections

# Resource constraints
MAX_TOTAL_AUGMENTATIONS = None  # No hard limit by default

# Paper-based augmentation strategies
AUGMENTATION_STRATEGY_ONE = {
    'rotation_range': 15,        # Random rotation -15° to 15°
    'horizontal_flip': True,     # Random horizontal mirroring
    'random_crop': True,         # Enable random cropping
    'crop_size': 96,             # Crop from 100x100 to 96x96
    'fill_mode': 'nearest'       # For handling pixels created by rotation
}

# Default is now paper-based approach
DEFAULT_AUGMENTATION_SETTINGS = AUGMENTATION_STRATEGY_ONE.copy()

# FER+ Emotion labels
FERPLUS_EMOTIONS = {
    0: 'neutral',
    1: 'happiness', 
    2: 'surprise',
    3: 'sadness',
    4: 'anger',
    5: 'disgust',
    6: 'fear',
    7: 'contempt'
}

# RAF-DB Emotion labels
RAFDB_EMOTIONS = {
    1: 'Surprise',
    2: 'Fear',
    3: 'Disgust',
    4: 'Happy',
    5: 'Sad',
    6: 'Angry', 
    7: 'Neutral'
}

# Age group mapping
AGE_GROUPS = {
    1: 'Child (0-3)',
    2: 'Teen (4-19)',
    3: 'Adult (20-39)',
    4: 'Middle-aged (40-69)',
    5: 'Senior (70+)'
}

# Gender mapping
GENDER_LABELS = {
    0: 'Male',
    1: 'Female'
}