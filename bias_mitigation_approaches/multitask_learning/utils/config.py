# bias_mitigation_approaches/multitask_learning/utils/config.py

import os

#-----------------------------------------------------
# DIRECTORY PATHS
#-----------------------------------------------------

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TRAINING_HISTORY_DIR = os.path.join(PLOTS_DIR, 'training_history')
FAIRNESS_TRENDS_DIR = os.path.join(PLOTS_DIR, 'fairness_trends')

# Ensure directories exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TRAINING_HISTORY_DIR, exist_ok=True)
os.makedirs(FAIRNESS_TRENDS_DIR, exist_ok=True)

# Dataset paths
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
MULTITASK_MODEL_OUTPUT_PATH = 'resources/models/multitask-finetuned-model.h5'

#-----------------------------------------------------
# MODEL PARAMETERS
#-----------------------------------------------------

# Model architecture
INPUT_SHAPE = (48, 48, 1)
NUM_EMOTION_CLASSES = 8  # FER+ has 8 emotion classes
NUM_GENDER_CLASSES = 2   # Male/Female
NUM_AGE_CLASSES = 5      # 5 age groups in RAF-DB
SPATIAL_DROPOUT_RATE = 0.2  # Increased from 0.15 for better generalization
LEAKY_RELU_SLOPE = 0.02
REGULARIZATION_RATE = 0.015  # Increased from 0.01 for better regularization
ATTENTION_HEADS = 4      # Number of attention heads for multi-head attention
SHARED_REPRESENTATION_SIZE = 128  # Size of shared representation layer

#-----------------------------------------------------
# MULTI-TASK PARAMETERS
#-----------------------------------------------------

# Loss weights for multi-task learning
EMOTION_LOSS_WEIGHT = 1.0      # Primary task weight
GENDER_LOSS_WEIGHT = 0.5       # Auxiliary task weight
AGE_LOSS_WEIGHT = 0.5          # Auxiliary task weight

# Fairness thresholds and parameters
FAIRNESS_THRESHOLD = 0.8       # Minimum acceptable fairness score
MAX_FAIRNESS_IMBALANCE = 0.2   # Maximum acceptable difference between group accuracies

#-----------------------------------------------------
# TRAINING HYPERPARAMETERS
#-----------------------------------------------------

# Training configuration
BATCH_SIZE = 64  # Smaller batch size for multi-task learning
VALIDATION_SPLIT = 0.15
MAX_EPOCHS = 1000  # Increased for better convergence with early stopping

# Learning rate settings
INITIAL_LR_PHASE1 = 5e-4  # Learning rate for first phase (training new heads)
INITIAL_LR_PHASE2 = 5e-6  # Learning rate for second phase (fine-tuning)
MIN_LEARNING_RATE = 1e-7  # Minimum learning rate

# Gradient clipping
GRADIENT_CLIP_NORM = 1.0  # Clip gradients to this norm value

# Learning rate warmup
WARMUP_EPOCHS_PHASE1 = 5  # Number of warmup epochs for phase 1
WARMUP_EPOCHS_PHASE2 = 3  # Number of warmup epochs for phase 2

# Training phases
PHASE1_EPOCHS = 20  # Number of epochs for initial phase (training new heads)

# Patience settings
EARLY_STOPPING_PATIENCE = 15
LR_PATIENCE = 7

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

# Mapping from RAF-DB to FER+ emotion indices
RAFDB_TO_FERPLUS = {
    1: 2,  # Surprise -> Surprise
    2: 6,  # Fear -> Fear
    3: 5,  # Disgust -> Disgust
    4: 1,  # Happy -> Happiness
    5: 3,  # Sad -> Sadness
    6: 4,  # Angry -> Anger
    7: 0   # Neutral -> Neutral
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