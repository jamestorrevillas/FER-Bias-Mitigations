# bias_mitigation_approaches/dynamic_weighting/utils/config.py

import os

#-----------------------------------------------------
# DIRECTORY PATHS
#-----------------------------------------------------

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Dataset paths
FER_DIR = 'resources/dataset/fer'
FER_CSV_PATH = os.path.join(FER_DIR, 'labels', 'fer2013.csv')
FERPLUS_CSV_PATH = os.path.join(FER_DIR, 'labels', 'fer2013new.csv')
DATASET_CSV_PATH = os.path.join(FER_DIR, 'labels', 'dataset.csv')
FER_AUGMENTED_DIR = os.path.join(FER_DIR, 'augmented')
FER_AUGMENTED_IMAGES_PATH = os.path.join(FER_AUGMENTED_DIR, 'augmented_images.npy')
FER_AUGMENTED_LABELS_PATH = os.path.join(FER_AUGMENTED_DIR, 'augmented_labels.npy')

RAFDB_DIR = 'resources/dataset/raf-db'
RAFDB_TRAIN_LABELS_PATH = os.path.join(RAFDB_DIR, 'labels', 'train_labels.csv')
RAFDB_TEST_LABELS_PATH = os.path.join(RAFDB_DIR, 'labels', 'test_labels.csv')
RAFDB_TRAIN_DIR = os.path.join(RAFDB_DIR, 'dataset', 'train')
RAFDB_TEST_DIR = os.path.join(RAFDB_DIR, 'dataset', 'test')
RAFDB_AUGMENTED_DIR = os.path.join(RAFDB_DIR, 'augmented')
RAFDB_AUGMENTED_IMAGES_PATH = os.path.join(RAFDB_AUGMENTED_DIR, 'augmented_images.npy')
RAFDB_AUGMENTED_LABELS_PATH = os.path.join(RAFDB_AUGMENTED_DIR, 'augmented_labels.npy')
RAFDB_AUGMENTED_DEMOGRAPHICS_PATH = os.path.join(RAFDB_AUGMENTED_DIR, 'augmented_demographics.pkl')

# Model paths
BASE_MODEL_PATH = 'resources/models/baseline-ferplus-model.h5'
DYNAMIC_MODEL_PATH = 'resources/models/dynamic-weighting-finetuned-model.h5'
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#-----------------------------------------------------
# MODEL PARAMETERS
#-----------------------------------------------------

# Model architecture (same as baseline)
INPUT_SHAPE = (48, 48, 1)
NUM_CLASSES = 8  # FER+ has 8 emotion classes
SPATIAL_DROPOUT_RATE = 0.10  # Same as original
LEAKY_RELU_SLOPE = 0.02  # Same as original
REGULARIZATION_RATE = 0.01  # Same as original

#-----------------------------------------------------
# TRAINING HYPERPARAMETERS
#-----------------------------------------------------

# Fine-tuning configuration
BATCH_SIZE = 64  # Smaller than original for more frequent feedback
VALIDATION_SPLIT = 0.15
MAX_FINE_TUNING_EPOCHS = 30

# Learning rate settings
INITIAL_LEARNING_RATE = 1e-4  # Lower than original for fine-tuning
MIN_LEARNING_RATE = 5e-6
LR_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 10

# Dynamic weighting parameters
FEEDBACK_FREQUENCY = 100  # Evaluate and update weights every N batches
WEIGHT_EVAL_EPOCHS = 1    # Number of epochs between full evaluations

# Weight adjustment parameters
INITIAL_WEIGHT_MULTIPLIER = 2.0
MAX_WEIGHT_MULTIPLIER = 5.0
WEIGHT_DECAY_FACTOR = 0.95    # Gradually reduce extreme weights
FAIRNESS_IMPROVEMENT_THRESHOLD = 0.01  # Minimum improvement to continue
MIN_GROUP_SAMPLES = 10  # Minimum samples needed for reliable evaluation

#-----------------------------------------------------
# MAPPINGS AND LABELS
#-----------------------------------------------------

# FER+ emotion mapping
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

# RAF-DB emotion mapping
RAFDB_EMOTIONS = {
    1: 'Surprise',
    2: 'Fear',
    3: 'Disgust',
    4: 'Happy',
    5: 'Sad',
    6: 'Angry',
    7: 'Neutral'
}

# Mapping from FER+ to RAF-DB emotion indices
FERPLUS_TO_RAFDB = {
    0: 7,  # neutral -> neutral
    1: 4,  # happiness -> happy
    2: 1,  # surprise -> surprise
    3: 5,  # sadness -> sad
    4: 6,  # anger -> angry
    5: 3,  # disgust -> disgust
    6: 2,  # fear -> fear
    7: None  # contempt (not in RAF-DB)
}

# Demographic mappings
GENDER_LABELS = {0: 'Male', 1: 'Female'}
AGE_GROUPS = {
    1: 'Child (0-12)',
    2: 'Teen (13-19)',
    3: 'Young Adult (20-40)',
    4: 'Adult (41-60)',
    5: 'Senior (60+)'
}

# Define emotion column names for dataset processing
EMOTION_COLUMNS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']