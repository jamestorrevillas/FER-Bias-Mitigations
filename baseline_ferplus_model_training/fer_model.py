# baseline_ferplus_model_training/fer_model.py

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import sys
import json
from tensorflow.python.client import device_lib

def check_gpu():
    """Check if GPU is available and configure TensorFlow accordingly"""
    print("Checking for GPU availability...")
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
            print("GPU is available and configured for use!")
            return True
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            return False
    else:
        print("No GPU found. Running on CPU...")
        return False

# Define base directory - modify this to your local project path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Goes up one level from script location
sys.path.append(base_dir)

# Paths - update to match your directory structure
dataset_dir = os.path.join(base_dir, 'resources', 'dataset', 'fer')
model_dir = os.path.join(base_dir, 'resources', 'models')
outputs_dir = os.path.join(base_dir, 'baseline_ferplus_model_training', 'outputs')
history_dir = os.path.join(outputs_dir, 'history')
plots_dir = os.path.join(outputs_dir, 'plots')

# Make sure these directories exist
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Import project modules
from data.dataset import get_dataset_dict
from data.model_class.DataPipelineParams import Augmentation, Dataset
from data.data import (get_data_pipeline, get_fer_class_mapping, 
                      get_fer_plus_class_mapping, get_labels, get_image_data)
from data.model_class.DataPipelineParams import DataPipelineParams
import predictions as ps
import models

## PARAMETERS
AUGMENTATION = Augmentation.HIGH
CROSS_ENTROPY = True
DATASET = Dataset.FERPLUS
N_CLASSES = 8
ORIGINAL_PREPROCESSING = True
USE_AUGMENTED_DATA = True  # Toggle flag to control augmented dataset usage

# Model file name
MODEL_NAME = 'augmented-baseline-ferplus-model'

## HYPERPARAMETERS
BATCH_SIZE = 128
EPOCHS = 1000
DROPOUT_RATE = 0.10
LEARNING_RATE = 0.005
LEAKY_RELU_SLOPE = 0.02
LR_PATIENCE = 20
PATIENCE = 30
REGULARIZATION_RATE = 0.01

def setup_model():
    """Initialize and compile the model"""
    model = models.get_performance_model(
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        dropout_rate=DROPOUT_RATE,
        regularization_rate=REGULARIZATION_RATE,
        n_classes=N_CLASSES,
        logits=True
    )
    
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def setup_callbacks():
    """Set up training callbacks"""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            min_delta=0.001
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=LR_PATIENCE,
            verbose=1,
            min_delta=0.001
        ),
        # Only keep one ModelCheckpoint
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, f"{MODEL_NAME}.h5"),
            monitor='val_accuracy',  # Monitor accuracy as the key metric
            save_best_only=True,
            verbose=1
        )
    ]

def save_history(history):
    """Save training history to JSON file"""
    history_dict = history.history
    for key in history_dict:
        history_dict[key] = [float(val) for val in history_dict[key]]
        
    # Save to history directory instead of model directory
    history_file_path = os.path.join(history_dir, f"{MODEL_NAME}_history.json")
    with open(history_file_path, 'w') as f:
        json.dump(history_dict, f)
    
    print(f"Training history saved to {history_file_path}")
    return history_dict

def evaluate_model(model, training_pipeline, validation_pipeline, test_pipeline):
    """Evaluate model on all datasets"""
    print("\nEvaluating on training set:")
    model.evaluate(training_pipeline)
    
    print("\nEvaluating on validation set:")
    model.evaluate(validation_pipeline)
    
    print("\nEvaluating on test set:")
    model.evaluate(test_pipeline)

def main():
    # Check for GPU
    using_gpu = check_gpu()
    
    print("\nLoading datasets...")
    # Load datasets
    try:
        dataset_dict = get_dataset_dict(
            dataset_dir=dataset_dir,
            fer_file_name='fer2013.csv',
            fer_plus_file_name='fer2013new.csv',
            use_augmented_data=USE_AUGMENTED_DATA
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure the dataset files exist in the following location:")
        print(f"- {os.path.join(dataset_dir, 'labels', 'fer2013.csv')}")
        print(f"- {os.path.join(dataset_dir, 'labels', 'fer2013new.csv')}")
        return
    
    print(f"Dataset loaded successfully. Training set size: {len(dataset_dict['train'])}")

    # Data pipeline parameters
    pipeline_params = DataPipelineParams(
        dataset=DATASET,
        cross_entropy=CROSS_ENTROPY,
        original_preprocessing=ORIGINAL_PREPROCESSING,
        batch_size=BATCH_SIZE,
        augmentation=AUGMENTATION,
        use_augmented_data=USE_AUGMENTED_DATA  # Pass the flag to pipeline
    )

    pipeline_params_test = DataPipelineParams(
        dataset=DATASET,
        original_preprocessing=ORIGINAL_PREPROCESSING,
        batch_size=BATCH_SIZE,
        augmentation=Augmentation.NONE,  # No augmentation for test/validation
        use_augmented_data=False  # Don't use augmented data for testing
    )

    print("Creating data pipelines...")
    # Create data pipelines
    training_pipeline = get_data_pipeline(
        dataset_df=dataset_dict['train'],
        params=pipeline_params,
        shuffle=True
    )
    
    validation_pipeline = get_data_pipeline(
        dataset_dict['valid'],
        params=pipeline_params_test
    )
    
    test_pipeline = get_data_pipeline(
        dataset_dict['test'],
        params=pipeline_params_test
    )

    print("Setting up model...")
    model = setup_model()
    callbacks = setup_callbacks()

    print("\nStarting training...")
    print(f"Training on {'GPU' if using_gpu else 'CPU'}")
    print(f"Using augmented data: {USE_AUGMENTED_DATA}")
    
    # Train the model
    history = model.fit(
        training_pipeline,
        validation_data=validation_pipeline,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    print("\nSaving training history...")
    history_dict = save_history(history)
    
    # Plot and save training history
    print("Generating training history plot...")
    ps.plot_training_history(history_dict, plots_dir)
    
    print("\nEvaluating model performance...")
    evaluate_model(model, training_pipeline, validation_pipeline, test_pipeline)
    
    print("\nGenerating confusion matrix...")
    # Generate confusion matrix
    y_pred = model.predict(test_pipeline, batch_size=BATCH_SIZE)
    y_pred = tf.nn.softmax(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = get_labels(dataset_dict['test'], params=pipeline_params_test)
    y_true = np.argmax(y_true, axis=1)
    
    confusion_mat = tf.math.confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    class_mapping = get_fer_plus_class_mapping() if DATASET == Dataset.FERPLUS else get_fer_class_mapping()
    class_names = list(class_mapping.values())
    ps.plot_confusion_matrix(confusion_mat, class_names, plots_dir)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()