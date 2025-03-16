# Facial Emotion Recognition Bias Mitigation - Data Augmentation Approach

This module implements a data augmentation-based approach to mitigate biases in facial emotion recognition systems. It consists of two complementary strategies: emotion-based augmentation and demographic-based augmentation, both aimed at improving fairness while preserving overall performance.

## Table of Contents

- [Overview](#overview)
- [Technical Approach](#technical-approach)
- [Implementation Details](#implementation-details)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Known Issues and Limitations](#known-issues-and-limitations)
- [Future Improvements](#future-improvements)

## Overview

Facial emotion recognition systems often exhibit biases toward certain demographic groups and emotion categories. This project addresses these biases by:

1. **Emotion-based augmentation**: Balances recognition accuracy across emotion categories by augmenting underrepresented emotions in the FER+ dataset.

2. **Demographic-based augmentation**: Improves fairness across demographic attributes (age, gender) by supplementing the RAF-DB dataset with demographically-targeted augmentations.

The implementation uses a fine-tuning approach with carefully calibrated parameters to prevent catastrophic forgetting while allowing adaptation to new data distributions.

## Technical Approach

### Dual-Domain Data Augmentation

Our approach employs a dual-domain strategy:

- **Emotion domain**: Targets emotion categories with poor recognition rates or low representation
- **Demographic domain**: Focuses on demographic groups with performance disparities

### Smart Fine-Tuning Architecture

The fine-tuning process employs these key techniques:

1. **Layer freezing**: Preserves early feature extraction layers while allowing adaptation in final layers
2. **Gradient clipping**: Prevents extreme weight updates that could destroy pre-trained knowledge
3. **Controlled learning rates**: Uses conservative learning rates with gradual warmup
4. **Stability-focused regularization**: Employs increased dropout and L2 regularization

## Implementation Details

### Data Pipeline

The implementation processes two distinct datasets:

1. **FER+ Dataset**: Used for emotion-based augmentation
   - 8 emotion categories (neutral, happiness, surprise, sadness, anger, disgust, fear, contempt)
   - Contains ~36K grayscale 48×48 facial images

2. **RAF-DB Dataset**: Used for demographic-based augmentation
   - 7 emotion categories with demographic metadata (age, gender)
   - Contains ~12K more diverse facial images with demographic labels

### Augmentation Strategy

The augmentation process for both domains:

1. Analyzes class distribution to identify underrepresented classes
2. Applies targeted data augmentation with appropriate transformations
3. Combines original and augmented data for fine-tuning
4. Uses specialized strategies for severe imbalances

### Model Architecture

The implementation fine-tunes a pre-trained FER+ baseline model:

- CNN architecture with separable convolutions
- ~170K trainable parameters
- 8 emotion output classes
- Structured to enable selective layer freezing

## Configuration

The system is configured through `config.py` with the following key parameters:

### Layer Freezing Settings

```python
FREEZE_LAYERS = True           # Enable layer freezing
NUM_TRAINABLE_LAYERS = 6       # Allow final layers to be trainable
FREEZABLE_LAYER_TYPES = [      # Types of layers to freeze
    'SeparableConv2D',
    'Conv2D',
    'BatchNormalization'
]
```

### Learning Rate Settings

```python
INITIAL_LEARNING_RATE = 1e-5   # 30x smaller than original
TARGET_LEARNING_RATE = 3e-5    # Still conservative but allows adaptation
MIN_LEARNING_RATE = 1e-6       # Reasonable lower bound
WARMUP_EPOCHS = 8              # Gradual warmup phase
```

### Regularization Settings

```python
SPATIAL_DROPOUT_RATE = 0.20    # Moderate increase from 0.15
REGULARIZATION_RATE = 0.0125   # 25% increase from original 0.01
```

### Training Configuration

```python
BATCH_SIZE = 64                # Half of original for more stable updates
MAX_EPOCHS = 100               # Realistic upper limit
LR_PATIENCE = 15               # Moderately patient learning rate scheduler
EARLY_STOPPING_PATIENCE = 30   # Balanced patience for convergence
```

## Running the Code

### Prerequisites

- Python 3.8+
- TensorFlow 2.6+
- Required datasets:
  - FER+ dataset (fer2013.csv and fer2013new.csv)
  - RAF-DB dataset with demographic annotations

### Running Emotion-Based Augmentation

```bash
# Navigate to the project directory
cd bias_mitigation_approaches/data_augmentation

# Run emotion augmentation
python emotion_augmentation.py
```

### Running Demographic-Based Augmentation

```bash
# Navigate to the project directory
cd bias_mitigation_approaches/data_augmentation

# Run demographic augmentation
python demographic_augmentation.py
```

### Evaluating Results

```bash
# Run comparative analysis
cd utils/comparative_analysis
python analyze.py
```

## Known Issues and Limitations

1. **Catastrophic Forgetting**: Fine-tuning may lose knowledge from pre-trained models despite mitigation strategies
2. **Domain Adaptation Gap**: Performance differences between FER+ and RAF-DB datasets require careful handling
3. **Limited Dataset Size**: The effectiveness is constrained by the relatively small size of demographically-labeled datasets
4. **Computational Requirements**: Fine-tuning with the current configuration can be compute-intensive
5. **Class Imbalance**: Extreme imbalances in emotion categories (e.g., disgust, contempt) remain challenging

## Future Improvements

1. **Knowledge Distillation**: Incorporate distillation losses to better preserve baseline model knowledge
2. **More Advanced Augmentation**: Explore GAN-based or style transfer approaches for higher quality augmentations
3. **Multi-Task Learning**: Integrate demographic prediction as auxiliary tasks during fine-tuning
4. **Cross-Dataset Validation**: Improve robustness by validating across multiple benchmark datasets
5. **Ensemble Approach**: Combine multiple bias mitigation strategies for more robust results
