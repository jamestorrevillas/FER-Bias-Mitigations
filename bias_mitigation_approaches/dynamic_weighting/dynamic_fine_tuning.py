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
os.makedirs(config.LOGS_DIR, exist_ok=True)

def load_pretrained_model(model_path=config.BASE_MODEL_PATH):
    """Load pre-trained FER+ model"""
    print(f"Loading pre-trained model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    
    return model

def setup_datasets(train_images, train_labels, val_images, val_labels, 
                   rafdb_val_images, rafdb_val_labels, rafdb_val_demographic_info,
                   rafdb_test_images, rafdb_test_labels, rafdb_test_demographic_info,
                   train_demographic_info=None):
    """Setup datasets for training and evaluation"""
    print("\nPreparing datasets...")
    
    # Create validation dataset
    if config.USE_EFFICIENT_DATASET:
        validation_dataset = data_loader.create_efficient_dataset(
            val_images, val_labels, batch_size=config.BATCH_SIZE
        )
    else:
        val_labels_onehot = tf.keras.utils.to_categorical(val_labels, config.NUM_CLASSES)
        validation_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels_onehot))
        validation_dataset = validation_dataset.batch(config.BATCH_SIZE)
    
    # Create RAF-DB datasets for validation and testing
    rafdb_val_dataset = (rafdb_val_images, rafdb_val_labels, rafdb_val_demographic_info)
    rafdb_test_dataset = (rafdb_test_images, rafdb_test_labels, rafdb_test_demographic_info)
    
    print(f"Training set: {len(train_images)} samples")
    print(f"Validation set: {len(val_images)} samples")
    print(f"RAF-DB validation set: {len(rafdb_val_images)} samples")
    print(f"RAF-DB test set: {len(rafdb_test_images)} samples")
    
    return train_images, train_labels, validation_dataset, rafdb_val_dataset, rafdb_test_dataset, train_demographic_info

def dynamic_fine_tuning(model, train_images, train_labels, validation_dataset, 
                        rafdb_val_dataset, rafdb_test_dataset, train_demographic_info=None):
    """Fine-tune model with dynamic cross-dataset weight adjustment"""
    print("\nStarting dynamic fine-tuning...")
    
    # Unpack RAF-DB validation dataset
    rafdb_val_images, rafdb_val_labels, rafdb_val_demographic_info = rafdb_val_dataset
    
    # Unpack RAF-DB test dataset (only used for final evaluation)
    rafdb_test_images, rafdb_test_labels, rafdb_test_demographic_info = rafdb_test_dataset
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.INITIAL_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Initialize weight scheduler with demographic info if available
    weight_scheduler_instance = weight_scheduler.DynamicWeightScheduler(
        initial_weights=np.ones(len(train_labels)),
        emotion_labels=train_labels,
        demographic_info=train_demographic_info
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
        model, rafdb_val_images, rafdb_val_labels, rafdb_val_demographic_info
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
        if config.USE_EFFICIENT_DATASET:
            train_dataset = data_loader.create_efficient_dataset(
                train_images, 
                train_labels,
                weights=current_weights,
                batch_size=config.BATCH_SIZE,
                shuffle_buffer=config.SHUFFLE_BUFFER_SIZE
            )
        else:
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
        
        # Evaluate fairness on RAF-DB validation dataset
        fairness_metrics = metrics.evaluate_model_fairness(
            model, rafdb_val_images, rafdb_val_labels, rafdb_val_demographic_info
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
        
        # Calculate combined fairness score
        combined_fairness = (gender_fairness + age_fairness + emotion_fairness) / 3
        fairness_improvement = combined_fairness - best_fairness_score
        
        # Check for improvement in fairness (primary criterion)
        if fairness_improvement > config.FAIRNESS_IMPROVEMENT_THRESHOLD:
            best_fairness_score = combined_fairness
            patience_counter = 0
            
            # Save best model
            model.save(config.DYNAMIC_MODEL_PATH)
            print(f"New best model saved with fairness score: {best_fairness_score:.4f}")
            print(f"Gender: {gender_fairness:.4f}, Age: {age_fairness:.4f}, Emotion: {emotion_fairness:.4f}")
            
            # Save metrics
            with open(os.path.join(config.RESULTS_DIR, 'best_metrics.json'), 'w') as f:
                json.dump({
                    'val_loss': float(val_loss),
                    'val_accuracy': float(val_accuracy),
                    'gender_fairness': float(gender_fairness),
                    'age_fairness': float(age_fairness),
                    'emotion_fairness': float(emotion_fairness),
                    'overall_accuracy': float(overall_accuracy),
                    'combined_fairness': float(combined_fairness)
                }, f, indent=4)
        # Check for improvement in validation loss (secondary criterion)
        elif val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss improved to {best_val_loss:.4f}, but fairness did not improve sufficiently")
            # Don't reset patience counter for val_loss improvements
            
            # Save checkpoint for best validation loss
            loss_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_val_loss_model.h5")
            model.save(loss_checkpoint_path)
        else:
            patience_counter += 1
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.h5")
            model.save(checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        
        # Check if we should stop training
        if patience_counter >= config.FAIRNESS_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs with no fairness improvement")
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
    
    # Final fairness evaluation on test set
    print("\nPerforming final evaluation on RAF-DB test set...")
    final_metrics = metrics.evaluate_model_fairness(
        best_model, rafdb_test_images, rafdb_test_labels, rafdb_test_demographic_info
    )
    
    # Print and save final test set results
    print("\nFinal Test Set Results:")
    print(f"Gender Fairness: {final_metrics['gender_metrics']['fairness_score']:.4f}")
    print(f"Age Fairness: {final_metrics['age_metrics']['fairness_score']:.4f}")
    print(f"Emotion Fairness: {final_metrics['emotion_fairness']:.4f}")
    print(f"Overall Accuracy: {final_metrics['overall_accuracy']:.4f}")
    
    # Save final test metrics
    with open(os.path.join(config.RESULTS_DIR, 'final_test_metrics.json'), 'w') as f:
        json.dump({
            'gender_fairness': float(final_metrics['gender_metrics']['fairness_score']),
            'age_fairness': float(final_metrics['age_metrics']['fairness_score']),
            'emotion_fairness': float(final_metrics['emotion_fairness']),
            'overall_accuracy': float(final_metrics['overall_accuracy'])
        }, f, indent=4)
    
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
    
    return best_model, fairness_history, weight_history, metrics_history, final_metrics

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
    print(f"Feedback frequency: every epoch")
    print(f"Using efficient dataset: {config.USE_EFFICIENT_DATASET}")
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
    
    # 3. Split RAF-DB test set into validation and test portions
    print("\nSplitting RAF-DB test set into validation and test portions...")
    rafdb_val_images, rafdb_test_images, rafdb_val_labels, rafdb_test_labels = train_test_split(
        rafdb_test_images, rafdb_test_labels, test_size=0.5, random_state=42, stratify=rafdb_test_labels
    )
    
    # Split demographic info accordingly
    rafdb_val_demographic_info = rafdb_test_demographic_info[:len(rafdb_val_labels)]
    rafdb_test_demographic_info = rafdb_test_demographic_info[len(rafdb_val_labels):]
    
    print(f"RAF-DB validation set: {len(rafdb_val_images)} samples")
    print(f"RAF-DB test set: {len(rafdb_test_images)} samples")
    
    # 4. Combine original and augmented FER+ data
    print("\nCombining original and augmented FER+ data...")
    X_train, y_train = data_loader.prepare_combined_data(
        original_train_images, 
        original_train_labels, 
        augmented_images, 
        augmented_labels
    )
    
    # 5. Generate demographic proxy information for FER+ dataset
    # This is used for intersectional weight adjustment
    print("\nGenerating demographic proxy information for FER+ dataset...")
    train_demographic_info = data_loader.extract_fer_demographic_proxy(
        X_train, 
        y_train,
        rafdb_train_demographic_info
    )
    
    # 6. Create train/validation split
    print("\nCreating train/validation split...")
    train_images, val_images, train_labels, val_labels, train_indices, val_indices = train_test_split(
        X_train, y_train, np.arange(len(X_train)),
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=y_train
    )
    
    # Split demographic info according to indices
    train_demographic_info = [train_demographic_info[i] for i in train_indices]
    
    # 7. Load pre-trained model
    model = load_pretrained_model()
    
    # 8. Setup datasets
    train_images, train_labels, validation_dataset, rafdb_val_dataset, rafdb_test_dataset, train_demographic_info = setup_datasets(
        train_images, train_labels, val_images, val_labels,
        rafdb_val_images, rafdb_val_labels, rafdb_val_demographic_info,
        rafdb_test_images, rafdb_test_labels, rafdb_test_demographic_info,
        train_demographic_info
    )
    
    # 9. Perform dynamic fine-tuning
    best_model, fairness_history, weight_history, metrics_history, final_metrics = dynamic_fine_tuning(
        model, train_images, train_labels, validation_dataset,
        rafdb_val_dataset, rafdb_test_dataset, train_demographic_info
    )
    
    # 10. Final evaluation on FER+ test set
    print("\nEvaluating best model on FER+ test set...")
    test_labels_onehot = tf.keras.utils.to_categorical(test_labels, config.NUM_CLASSES)
    test_loss, test_accuracy = best_model.evaluate(test_images, test_labels_onehot, verbose=1)
    print(f"FER+ Test Loss: {test_loss:.4f}")
    print(f"FER+ Test Accuracy: {test_accuracy:.4f}")
    
    # Save FER+ test results
    with open(os.path.join(config.RESULTS_DIR, 'ferplus_test_metrics.json'), 'w') as f:
        json.dump({
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy)
        }, f, indent=4)
    
    # 11. Summary of fairness improvements
    initial_gender_fairness = fairness_history['gender_fairness'][0]
    final_gender_fairness = final_metrics['gender_metrics']['fairness_score']
    
    initial_age_fairness = fairness_history['age_fairness'][0]
    final_age_fairness = final_metrics['age_metrics']['fairness_score']
    
    initial_emotion_fairness = fairness_history['emotion_fairness'][0]
    final_emotion_fairness = final_metrics['emotion_fairness']
    
    initial_accuracy = fairness_history['overall_accuracy'][0]
    final_accuracy = final_metrics['overall_accuracy']
    
    # Calculate improvements
    gender_improvement = (final_gender_fairness - initial_gender_fairness) * 100
    age_improvement = (final_age_fairness - initial_age_fairness) * 100
    emotion_improvement = (final_emotion_fairness - initial_emotion_fairness) * 100
    accuracy_change = (final_accuracy - initial_accuracy) * 100
    
    # Print summary
    print("\n" + "=" * 80)
    print("FAIRNESS IMPROVEMENT SUMMARY")
    print("=" * 80)
    print(f"Gender Fairness:   {initial_gender_fairness:.4f} -> {final_gender_fairness:.4f} ({gender_improvement:+.2f}%)")
    print(f"Age Fairness:      {initial_age_fairness:.4f} -> {final_age_fairness:.4f} ({age_improvement:+.2f}%)")
    print(f"Emotion Fairness:  {initial_emotion_fairness:.4f} -> {final_emotion_fairness:.4f} ({emotion_improvement:+.2f}%)")
    print(f"Overall Accuracy:  {initial_accuracy:.4f} -> {final_accuracy:.4f} ({accuracy_change:+.2f}%)")
    print("=" * 80)

    # Save summary to file
    with open(os.path.join(config.RESULTS_DIR, 'improvement_summary.txt'), 'w') as f:
        f.write("FAIRNESS IMPROVEMENT SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Gender Fairness:   {initial_gender_fairness:.4f} -> {final_gender_fairness:.4f} ({gender_improvement:+.2f}%)\n")
        f.write(f"Age Fairness:      {initial_age_fairness:.4f} -> {final_age_fairness:.4f} ({age_improvement:+.2f}%)\n")
        f.write(f"Emotion Fairness:  {initial_emotion_fairness:.4f} -> {final_emotion_fairness:.4f} ({emotion_improvement:+.2f}%)\n")
        f.write(f"Overall Accuracy:  {initial_accuracy:.4f} -> {final_accuracy:.4f} ({accuracy_change:+.2f}%)\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nFine-tuning complete!")
    print(f"Results saved to {config.RESULTS_DIR}")
    print(f"Fine-tuned model saved to {config.DYNAMIC_MODEL_PATH}")
    
    # Return the metrics for use in hyperparameter tuning
    return {
        'gender_fairness': float(final_gender_fairness),
        'age_fairness': float(final_age_fairness),
        'emotion_fairness': float(final_emotion_fairness),
        'overall_accuracy': float(final_accuracy),
        'combined_fairness': float((final_gender_fairness + final_age_fairness + final_emotion_fairness) / 3),
        'ferplus_accuracy': float(test_accuracy)
    }

if __name__ == "__main__":
    main()