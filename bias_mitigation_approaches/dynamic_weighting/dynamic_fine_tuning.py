# bias_mitigation_approaches/dynamic_weighting/dynamic_fine_tuning.py

"""
This script implements dynamic cross-dataset weight adjustment for facial emotion recognition.
It loads a pre-trained FER+ model and fine-tunes it using a feedback loop between
the FER+ dataset (for training) and the RAF-DB dataset (for demographic fairness evaluation).

Usage:
    python dynamic_fine_tuning.py

Output:
    - Fine-tuned model saved to the path specified in config
    - Evaluation metrics and visualizations in results directory
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import time
import json

# Import our utility modules
from utils import config
from utils import data_loader
from utils import metrics
from utils import visualization
from utils import weight_scheduler

# Set up GPU memory growth for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using {len(gpus)} GPU(s)")
else:
    print("Using CPU")

# Create output directories
os.makedirs(config.PLOTS_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

def load_pretrained_model(model_path=config.BASE_MODEL_PATH):
    """Load pre-trained FER+ model"""
    print(f"Loading pre-trained model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    
    return model

def setup_datasets(train_images, train_labels, val_images, val_labels, rafdb_images, rafdb_labels, rafdb_demographic_info):
    """Setup datasets for training and evaluation"""
    print("\nPreparing datasets...")
    
    # Create validation dataset
    val_labels_onehot = tf.keras.utils.to_categorical(val_labels, config.NUM_CLASSES)
    validation_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_onehot))
    validation_dataset = validation_dataset.batch(config.BATCH_SIZE)
    
    # Create RAF-DB evaluation dataset
    rafdb_dataset = (rafdb_images, rafdb_labels, rafdb_demographic_info)
    
    print(f"Training set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")
    print(f"RAF-DB evaluation set: {len(rafdb_images)} samples")
    
    return train_images, train_labels, validation_dataset, rafdb_dataset

def dynamic_fine_tuning(model, train_images, train_labels, validation_dataset, rafdb_dataset):
    """Fine-tune model with dynamic cross-dataset weight adjustment"""
    print("\nStarting dynamic fine-tuning...")
    
    # Unpack RAF-DB dataset
    rafdb_images, rafdb_labels, rafdb_demographic_info = rafdb_dataset
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.INITIAL_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Initialize weight scheduler
    weight_scheduler_instance = weight_scheduler.DynamicWeightScheduler(
        initial_weights=np.ones(len(train_labels)),
        emotion_labels=train_labels
    )
    
    # Setup tracking variables
    best_val_loss = float('inf')
    best_fairness_score = 0.0
    patience_counter = 0
    fairness_history = {
        'gender_fairness': [],
        'age_fairness': [],
        'emotion_fairness': [],
        'overall_accuracy': []
    }
    weight_history = []
    metrics_history = []
    
    # Create log file
    log_file = os.path.join(config.LOGS_DIR, 'dynamic_fine_tuning.log')
    
    # Initial fairness evaluation
    print("\nPerforming initial fairness evaluation...")
    fairness_metrics = metrics.evaluate_model_fairness(
        model, rafdb_images, rafdb_labels, rafdb_demographic_info
    )
    
    # Log initial fairness metrics
    with open(log_file, 'w') as f:
        f.write(f"Initial fairness metrics:\n")
        f.write(f"Gender fairness: {fairness_metrics['gender_metrics']['fairness_score']:.4f}\n")
        f.write(f"Age fairness: {fairness_metrics['age_metrics']['fairness_score']:.4f}\n")
        f.write(f"Emotion fairness: {fairness_metrics['emotion_fairness']:.4f}\n")
        f.write(f"Overall accuracy: {fairness_metrics['overall_accuracy']:.4f}\n\n")
    
    # Track initial metrics
    fairness_history['gender_fairness'].append(fairness_metrics['gender_metrics']['fairness_score'])
    fairness_history['age_fairness'].append(fairness_metrics['age_metrics']['fairness_score'])
    fairness_history['emotion_fairness'].append(fairness_metrics['emotion_fairness'])
    fairness_history['overall_accuracy'].append(fairness_metrics['overall_accuracy'])
    metrics_history.append(fairness_metrics)
    
    # Initial weight distribution
    weight_distribution = metrics.compute_weight_distribution(weight_scheduler_instance.get_current_weights())
    weight_history.append(weight_distribution)
    
    # Main training loop
    for epoch in range(config.MAX_FINE_TUNING_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.MAX_FINE_TUNING_EPOCHS}")
        start_time = time.time()
        
        # Update sample weights based on fairness metrics
        current_weights = weight_scheduler_instance.update_weights(fairness_metrics)
        
        # Create weighted dataset for this epoch
        train_labels_onehot = tf.keras.utils.to_categorical(train_labels, config.NUM_CLASSES)
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels_onehot, current_weights)
        )
        train_dataset = train_dataset.shuffle(len(train_images)).batch(config.BATCH_SIZE)
        
        # Train for one epoch
        history = model.fit(
            train_dataset,
            epochs=1,
            validation_data=validation_dataset,
            verbose=1
        )
        
        # Evaluate fairness on RAF-DB dataset
        fairness_metrics = metrics.evaluate_model_fairness(
            model, rafdb_images, rafdb_labels, rafdb_demographic_info
        )
        
        # Track metrics
        val_loss = history.history['val_loss'][0]
        val_accuracy = history.history['val_accuracy'][0]
        gender_fairness = fairness_metrics['gender_metrics']['fairness_score']
        age_fairness = fairness_metrics['age_metrics']['fairness_score']
        emotion_fairness = fairness_metrics['emotion_fairness']
        overall_accuracy = fairness_metrics['overall_accuracy']
        
        fairness_history['gender_fairness'].append(gender_fairness)
        fairness_history['age_fairness'].append(age_fairness)
        fairness_history['emotion_fairness'].append(emotion_fairness)
        fairness_history['overall_accuracy'].append(overall_accuracy)
        metrics_history.append(fairness_metrics)
        
        # Track weight distribution
        weight_distribution = metrics.compute_weight_distribution(current_weights)
        weight_history.append(weight_distribution)
        
        # Calculate time
        epoch_time = time.time() - start_time
        
        # Log progress
        log_message = (
            f"Epoch {epoch+1}/{config.MAX_FINE_TUNING_EPOCHS} - {epoch_time:.2f}s - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_accuracy:.4f} - "
            f"gender_fairness: {gender_fairness:.4f} - age_fairness: {age_fairness:.4f} - "
            f"emotion_fairness: {emotion_fairness:.4f} - accuracy: {overall_accuracy:.4f}"
        )
        print(log_message)
        
        with open(log_file, 'a') as f:
            f.write(log_message + "\n")
        
        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model.save(config.DYNAMIC_MODEL_PATH)
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")
            
            # Save metrics
            with open(os.path.join(config.RESULTS_DIR, 'best_metrics.json'), 'w') as f:
                json.dump({
                    'val_loss': float(val_loss),
                    'val_accuracy': float(val_accuracy),
                    'gender_fairness': float(gender_fairness),
                    'age_fairness': float(age_fairness),
                    'emotion_fairness': float(emotion_fairness),
                    'overall_accuracy': float(overall_accuracy)
                }, f, indent=4)
        else:
            patience_counter += 1
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.h5")
            model.save(checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Check if we should stop training
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training complete
    print("\nTraining complete!")
    
    # Save fairness history
    with open(os.path.join(config.RESULTS_DIR, 'fairness_history.pkl'), 'wb') as f:
        pickle.dump(fairness_history, f)
    
    # Save weight history
    with open(os.path.join(config.RESULTS_DIR, 'weight_history.pkl'), 'wb') as f:
        pickle.dump(weight_history, f)
    
    # Save metrics history
    with open(os.path.join(config.RESULTS_DIR, 'metrics_history.pkl'), 'wb') as f:
        pickle.dump(metrics_history, f)
    
    # Create visualizations
    visualization.plot_fairness_trends(
        fairness_history,
        os.path.join(config.PLOTS_DIR, 'fairness_trends.png')
    )
    
    visualization.plot_weight_evolution(
        weight_history,
        os.path.join(config.PLOTS_DIR, 'weight_evolution.png')
    )
    
    visualization.plot_group_performance(
        metrics_history, 
        'gender',
        os.path.join(config.PLOTS_DIR, 'gender_performance.png')
    )
    
    visualization.plot_group_performance(
        metrics_history, 
        'age',
        os.path.join(config.PLOTS_DIR, 'age_performance.png')
    )
    
    # Load best model for final evaluation
    best_model = tf.keras.models.load_model(config.DYNAMIC_MODEL_PATH)
    
    # Final fairness evaluation
    final_metrics = metrics.evaluate_model_fairness(
        best_model, rafdb_images, rafdb_labels, rafdb_demographic_info
    )
    
    # Create additional visualization for final performance
    visualization.plot_emotion_accuracies(
        final_metrics['emotion_accuracies'],
        'Final Emotion Recognition Accuracy',
        os.path.join(config.PLOTS_DIR, 'final_emotion_accuracies.png')
    )
    
    visualization.plot_intersectional_performance(
        final_metrics,
        os.path.join(config.PLOTS_DIR, 'intersectional_performance.png')
    )
    
    return best_model, fairness_history, weight_history, metrics_history

def main():
    # Print configuration information
    print("=" * 80)
    print("DYNAMIC CROSS-DATASET WEIGHTING FINE-TUNING")
    print("=" * 80)
    print(f"Base model: {config.BASE_MODEL_PATH}")
    print(f"Output model: {config.DYNAMIC_MODEL_PATH}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max epochs: {config.MAX_FINE_TUNING_EPOCHS}")
    print(f"Initial learning rate: {config.INITIAL_LEARNING_RATE}")
    print(f"Feedback frequency: every {config.FEEDBACK_FREQUENCY} batches")
    print("=" * 80)
    
    # 1. Load FER+ dataset (original + augmented for training)
    print("\nLoading FER+ datasets...")
    original_train_images, original_train_labels, test_images, test_labels = data_loader.load_ferplus_dataset()
    
    if original_train_images is None:
        print("Error loading original dataset. Exiting.")
        return
    
    # Load augmented FER+ dataset
    augmented_images, augmented_labels = data_loader.load_augmented_fer_dataset()
    
    # 2. Load RAF-DB dataset for evaluation
    print("\nLoading RAF-DB dataset...")
    rafdb_train_images, rafdb_train_labels, rafdb_train_demographic_info = data_loader.load_rafdb_dataset('train')
    rafdb_test_images, rafdb_test_labels, rafdb_test_demographic_info = data_loader.load_rafdb_dataset('test')
    
    if rafdb_test_images is None:
        print("Error loading RAF-DB dataset. Exiting.")
        return
    
    # 3. Combine original and augmented FER+ data
    print("\nCombining original and augmented FER+ data...")
    X_train, y_train = data_loader.prepare_combined_data(
        original_train_images, 
        original_train_labels, 
        augmented_images, 
        augmented_labels
    )
    
    # 4. Create train/validation split
    print("\nCreating train/validation split...")
    train_images, val_images, train_labels, val_labels = train_test_split(
        X_train, y_train,
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=y_train
    )
    
    # 5. Load pre-trained model
    model = load_pretrained_model()
    
    # 6. Setup datasets
    train_images, train_labels, validation_dataset, rafdb_dataset = setup_datasets(
        train_images, train_labels, val_images, val_labels,
        rafdb_test_images, rafdb_test_labels, rafdb_test_demographic_info
    )
    
    # 7. Perform dynamic fine-tuning
    best_model, fairness_history, weight_history, metrics_history = dynamic_fine_tuning(
        model, train_images, train_labels, validation_dataset, rafdb_dataset
    )
    
    # 8. Final evaluation on test set
    print("\nEvaluating best model on FER+ test set...")
    test_labels_onehot = tf.keras.utils.to_categorical(test_labels, config.NUM_CLASSES)
    test_loss, test_accuracy = best_model.evaluate(test_images, test_labels_onehot, verbose=1)
    print(f"FER+ Test Loss: {test_loss:.4f}")
    print(f"FER+ Test Accuracy: {test_accuracy:.4f}")
    
    # 9. Summary of fairness improvements
    initial_gender_fairness = fairness_history['gender_fairness'][0]
    final_gender_fairness = fairness_history['gender_fairness'][-1]
    
    initial_age_fairness = fairness_history['age_fairness'][0]
    final_age_fairness = fairness_history['age_fairness'][-1]
    
    initial_emotion_fairness = fairness_history['emotion_fairness'][0]
    final_emotion_fairness = fairness_history['emotion_fairness'][-1]
    
    initial_accuracy = fairness_history['overall_accuracy'][0]
    final_accuracy = fairness_history['overall_accuracy'][-1]
    
    # Calculate improvements
    gender_improvement = (final_gender_fairness - initial_gender_fairness) * 100
    age_improvement = (final_age_fairness - initial_age_fairness) * 100
    emotion_improvement = (final_emotion_fairness - initial_emotion_fairness) * 100
    accuracy_change = (final_accuracy - initial_accuracy) * 100
    
    # Print summary
    print("\n" + "=" * 80)
    print("FAIRNESS IMPROVEMENT SUMMARY")
    print("=" * 80)
    print(f"Gender Fairness:   {initial_gender_fairness:.4f} → {final_gender_fairness:.4f} ({gender_improvement:+.2f}%)")
    print(f"Age Fairness:      {initial_age_fairness:.4f} → {final_age_fairness:.4f} ({age_improvement:+.2f}%)")
    print(f"Emotion Fairness:  {initial_emotion_fairness:.4f} → {final_emotion_fairness:.4f} ({emotion_improvement:+.2f}%)")
    print(f"Overall Accuracy:  {initial_accuracy:.4f} → {final_accuracy:.4f} ({accuracy_change:+.2f}%)")
    print("=" * 80)
    
    # Save summary to file
    with open(os.path.join(config.RESULTS_DIR, 'improvement_summary.txt'), 'w') as f:
        f.write("FAIRNESS IMPROVEMENT SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Gender Fairness:   {initial_gender_fairness:.4f} → {final_gender_fairness:.4f} ({gender_improvement:+.2f}%)\n")
        f.write(f"Age Fairness:      {initial_age_fairness:.4f} → {final_age_fairness:.4f} ({age_improvement:+.2f}%)\n")
        f.write(f"Emotion Fairness:  {initial_emotion_fairness:.4f} → {final_emotion_fairness:.4f} ({emotion_improvement:+.2f}%)\n")
        f.write(f"Overall Accuracy:  {initial_accuracy:.4f} → {final_accuracy:.4f} ({accuracy_change:+.2f}%)\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nFine-tuning complete!")
    print(f"Results saved to {config.RESULTS_DIR}")
    print(f"Fine-tuned model saved to {config.DYNAMIC_MODEL_PATH}")

if __name__ == "__main__":
    main()