# bias_mitigation_approaches/multi_task_learning/model/multi_task_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers
import sys
import os

# Append parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def create_shared_representation_layer(features, regularization_rate=REGULARIZATION_RATE):
    """Create a shared representation layer with enhanced features"""
    
    # First expand the feature dimension to provide more capacity
    # This replaces the attention mechanism with a simpler approach
    expanded_features = layers.Dense(
        64,  # Expand to 64 dimensions first
        activation='relu',
        kernel_regularizer=regularizers.l2(regularization_rate),
        name='feature_expansion'
    )(features)
    
    # Add batch normalization for stability
    expanded_features = layers.BatchNormalization()(expanded_features)
    
    # Shared representation layer
    shared = layers.Dense(
        SHARED_REPRESENTATION_SIZE,
        activation='relu',
        kernel_regularizer=regularizers.l2(regularization_rate),
        name='shared_representation'
    )(expanded_features)
    
    # Add dropout for better generalization
    shared = layers.Dropout(SPATIAL_DROPOUT_RATE)(shared)
    
    return shared

def load_and_modify_base_model(base_model_path, num_emotion_classes=NUM_EMOTION_CLASSES):
    """
    Load pretrained FER+ model and prepare it for multi-task learning
    
    Args:
        base_model_path: Path to the pretrained FER+ model
        num_emotion_classes: Number of emotion classes (default: 8 for FER+)
        
    Returns:
        Model and feature extractor
    """
    # Load the base model
    base_model = tf.keras.models.load_model(base_model_path)
    
    # Print summary for debugging
    print("Base model summary:")
    base_model.summary()
    
    # Get the flattened features after GlobalAveragePooling
    feature_extractor = Model(
        inputs=base_model.input,
        outputs=base_model.layers[-1].output  # Get final layer output
    )
    
    print("Feature extractor output shape:", feature_extractor.output_shape)
    
    # Create multi-task model
    inputs = Input(shape=INPUT_SHAPE)
    features = feature_extractor(inputs)
    
    # Create shared representation layer
    shared_features = create_shared_representation_layer(features)
    
    # Task-specific layers with separate features for each task
    
    # Primary task: Emotion classification
    emotion_features = layers.Dense(
        64, 
        activation='relu',
        kernel_regularizer=regularizers.l2(REGULARIZATION_RATE),
        name='emotion_features'
    )(shared_features)
    emotion_features = layers.Dropout(SPATIAL_DROPOUT_RATE)(emotion_features)
    emotion_output = layers.Dense(num_emotion_classes, name='emotion_output')(emotion_features)
    
    # Auxiliary task 1: Gender classification with orthogonal regularization
    gender_features = layers.Dense(
        32, 
        activation='relu',
        kernel_regularizer=regularizers.l2(REGULARIZATION_RATE),
        name='gender_features'
    )(shared_features)
    gender_features = layers.Dropout(SPATIAL_DROPOUT_RATE)(gender_features)
    gender_output = layers.Dense(NUM_GENDER_CLASSES, name='gender_output')(gender_features)
    
    # Auxiliary task 2: Age classification
    age_features = layers.Dense(
        48, 
        activation='relu',
        kernel_regularizer=regularizers.l2(REGULARIZATION_RATE),
        name='age_features'
    )(shared_features)
    age_features = layers.Dropout(SPATIAL_DROPOUT_RATE)(age_features)
    age_output = layers.Dense(NUM_AGE_CLASSES, name='age_output')(age_features)
    
    # Create final model
    model = Model(
        inputs=inputs,
        outputs=[emotion_output, gender_output, age_output],
        name='multi_task_model'
    )
    
    return model, feature_extractor

def create_multi_task_model(base_model_path=BASE_MODEL_PATH):
    """
    Create multi-task learning model for emotion recognition with demographic awareness
    
    Args:
        base_model_path: Path to the pretrained FER+ model
        
    Returns:
        Multi-task model and feature extractor
    """
    # Load and modify base model
    model, feature_extractor = load_and_modify_base_model(base_model_path)
    
    # Return the model and feature extractor
    return model, feature_extractor