# bias_mitigation_approaches/data_augmentation/emotion_augmentation.py

"""
This script implements emotion-based bias mitigation for facial emotion recognition.
It loads a pre-trained FER+ model and fine-tunes it using augmented data from
the FER+ dataset to improve recognition of underrepresented emotion classes.

Usage:
    python emotion_augmentation.py

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

# Import our utility modules
from utils import config
from utils import data_loader
from utils import metrics
from utils import visualization

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
EMOTION_OUTPUT_DIR = os.path.join(config.RESULTS_DIR, 'emotion_augmentation')
os.makedirs(EMOTION_OUTPUT_DIR, exist_ok=True)

#-----------------------------------------------------
# CUSTOM CALLBACKS
#-----------------------------------------------------

class EmotionAccuracyCallback(tf.keras.callbacks.Callback):
    """Custom callback to track per-emotion accuracy during training"""
    
    def __init__(self, validation_data, emotion_mapping=None):
        super(EmotionAccuracyCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.emotion_mapping = emotion_mapping or config.FERPLUS_EMOTIONS
        
        # Convert one-hot y_val back to class indices for easier tracking
        self.y_val_indices = np.argmax(self.y_val, axis=1)
        
        # Initialize tracking dictionaries
        self.emotion_accuracies = {i: [] for i in range(len(self.emotion_mapping))}
        self.emotion_counts = np.bincount(self.y_val_indices, minlength=len(self.emotion_mapping))
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get predictions (logits)
        y_pred_logits = self.model.predict(self.X_val, verbose=0)
        
        # Apply softmax to get probabilities
        y_pred_probs = metrics.softmax(y_pred_logits)
        
        # Get predicted classes
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Calculate per-emotion accuracy
        for emotion_idx in range(len(self.emotion_mapping)):
            # Find samples of this emotion
            mask = self.y_val_indices == emotion_idx
            if np.sum(mask) > 0:  # Avoid division by zero
                # Calculate accuracy for this emotion
                accuracy = np.mean(y_pred[mask] == emotion_idx)
                self.emotion_accuracies[emotion_idx].append(accuracy)
            else:
                self.emotion_accuracies[emotion_idx].append(0)
                emotion_name = self.emotion_mapping[emotion_idx]

class WarmupScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler with warmup phase"""
    
    def __init__(self, warmup_epochs=config.WARMUP_EPOCHS,
                initial_lr=config.INITIAL_LEARNING_RATE,
                target_lr=config.TARGET_LEARNING_RATE):
        super(WarmupScheduler, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            print(f"\nEpoch {epoch+1}: Learning rate set to {lr:.2e}")

#-----------------------------------------------------
# MODEL LOADING AND TRAINING
#-----------------------------------------------------

def load_pretrained_model(model_path=config.BASE_MODEL_PATH):
    """Load pre-trained FER+ model with layer freezing"""
    print(f"Loading pre-trained model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
    
    # Apply layer freezing if enabled in config
    if config.FREEZE_LAYERS:
        # Count layers to determine which ones to freeze
        total_layers = len(model.layers)
        freeze_until = total_layers - config.NUM_TRAINABLE_LAYERS
        
        # Freeze early layers
        frozen_count = 0
        for i, layer in enumerate(model.layers):
            # Only freeze if it's before our cutoff and of a freezable type
            layer_type = layer.__class__.__name__
            if (i < freeze_until and 
                any(freezable_type in layer_type for freezable_type in config.FREEZABLE_LAYER_TYPES)):
                layer.trainable = False
                frozen_count += 1
                print(f"Freezing layer {i}: {layer.name} ({layer_type})")
            else:
                print(f"Keeping layer {i} trainable: {layer.name} ({layer_type})")
        
        print(f"Froze {frozen_count} layers, keeping {total_layers - frozen_count} layers trainable")
    
    return model

def fine_tune_model(model, X_train, y_train, X_val, y_val):
    """Fine-tune the model with augmented data"""
    print("\nPreparing for fine-tuning...")
    
    # Convert integer labels to one-hot format
    y_train_onehot = tf.keras.utils.to_categorical(y_train, config.NUM_CLASSES)
    y_val_onehot = tf.keras.utils.to_categorical(y_val, config.NUM_CLASSES)

    # Compile the model with gradient clipping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.INITIAL_LEARNING_RATE,
            clipnorm=1.0  # Add gradient clipping for stability
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Initialize emotion accuracy callback
    emotion_callback = EmotionAccuracyCallback(
        validation_data=(X_val, y_val_onehot),
        emotion_mapping=config.FERPLUS_EMOTIONS
    )
    
    # Set up callbacks
    callbacks = [
        WarmupScheduler(
            warmup_epochs=config.WARMUP_EPOCHS,
            initial_lr=config.INITIAL_LEARNING_RATE,
            target_lr=config.TARGET_LEARNING_RATE
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1
        ),
        # Single ModelCheckpoint that only saves the best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.EMOTION_MODEL_OUTPUT_PATH,
            monitor='val_loss',
            save_best_only=True,  # Only save when validation loss improves
            mode='min',          # Lower is better for loss
            verbose=1
        ),
        emotion_callback
    ]
    
    # Start fine-tuning
    print(f"\nStarting fine-tuning with batch size {config.BATCH_SIZE}...")
    history = model.fit(
        X_train, y_train_onehot,
        validation_data=(X_val, y_val_onehot),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )
    
    print(f"Fine-tuning complete. Best model saved to {config.EMOTION_MODEL_OUTPUT_PATH}")
    
    return model, history, emotion_callback

def plot_emotion_accuracy_trends(emotion_callback, output_dir=EMOTION_OUTPUT_DIR):
    """Plot per-emotion accuracy trends during training"""
    plt.figure(figsize=(14, 8))
    epochs = range(1, len(next(iter(emotion_callback.emotion_accuracies.values()))) + 1)
    
    for emotion_idx, accuracies in emotion_callback.emotion_accuracies.items():
        if emotion_idx in config.FERPLUS_EMOTIONS:
            emotion_name = config.FERPLUS_EMOTIONS[emotion_idx]
            plt.plot(epochs, accuracies, marker='o', linestyle='-', label=f'{emotion_name}')
    
    plt.title('Validation Accuracy by Emotion Class')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'emotion_accuracy_trends.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Also save as CSV for further analysis
    emotion_accuracy_df = pd.DataFrame({
        'epoch': epochs
    })
    
    for emotion_idx, accuracies in emotion_callback.emotion_accuracies.items():
        if emotion_idx in config.FERPLUS_EMOTIONS:
            emotion_name = config.FERPLUS_EMOTIONS[emotion_idx]
            emotion_accuracy_df[emotion_name] = accuracies
    
    csv_path = os.path.join(output_dir, 'emotion_accuracy_trends.csv')
    emotion_accuracy_df.to_csv(csv_path, index=False)
    print(f"Emotion accuracy trends saved to {output_dir}")

#-----------------------------------------------------
# VISUALIZATION FUNCTIONS
#-----------------------------------------------------

def visualize_augmentation_examples(augmented_images, augmented_labels, original_images=None, original_labels=None, output_dir=EMOTION_OUTPUT_DIR):
    """Visualize examples of augmented samples for each emotion"""
    # Display augmented samples
    if augmented_images is not None and len(augmented_images) > 0:
        # If original images are provided, show side-by-side comparisons
        if original_images is not None and original_labels is not None:
            fig, axes = plt.subplots(4, 4, figsize=(15, 12))
            axes = axes.flatten()
            
            # Find unique emotions
            unique_labels = np.unique(original_labels)
            displayed_count = 0
            
            for label in unique_labels:
                if displayed_count >= 8:  # Only show examples for 8 emotions
                    break
                    
                # Find original samples of this emotion
                orig_mask = original_labels == label
                orig_indices = np.where(orig_mask)[0]
                
                # Find augmented samples of this emotion
                aug_mask = augmented_labels == label
                aug_indices = np.where(aug_mask)[0]
                
                if len(orig_indices) > 0 and len(aug_indices) > 0:
                    # Show original
                    orig_idx = orig_indices[0]
                    axes[displayed_count*2].imshow(original_images[orig_idx].squeeze(), cmap='gray')
                    axes[displayed_count*2].set_title(f"Original {config.FERPLUS_EMOTIONS.get(label, label)}")
                    axes[displayed_count*2].axis('off')
                    
                    # Show augmented
                    aug_idx = aug_indices[0]
                    axes[displayed_count*2+1].imshow(augmented_images[aug_idx].squeeze(), cmap='gray')
                    axes[displayed_count*2+1].set_title(f"Augmented {config.FERPLUS_EMOTIONS.get(label, label)}")
                    axes[displayed_count*2+1].axis('off')
                    
                    displayed_count += 1
            
            # Hide any unused axes
            for i in range(displayed_count*2, len(axes)):
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'augmentation_examples.png'))
            plt.close()
            
        # Otherwise just show augmented samples
        else:
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            axes = axes.flatten()
            
            # Get examples of each augmented emotion
            unique_labels = np.unique(augmented_labels)
            for i, label in enumerate(unique_labels):
                if i >= len(axes):
                    break
                    
                # Find samples with this label
                mask = augmented_labels == label
                idx = np.where(mask)[0]
                if len(idx) > 0:
                    sample_idx = idx[0]
                    img = augmented_images[sample_idx].squeeze()
                    axes[i].imshow(img, cmap='gray')
                    axes[i].set_title(f"{config.FERPLUS_EMOTIONS.get(label, label)}")
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'augmentation_examples.png'))
            plt.close()
            
        print(f"Augmentation examples visualization saved to {output_dir}")

#-----------------------------------------------------
# MAIN EXECUTION FLOW
#-----------------------------------------------------

def main():
    # Print configuration information
    print("=" * 80)
    print("EMOTION-BASED AUGMENTATION FINE-TUNING")
    print("=" * 80)
    print(f"Base model: {config.BASE_MODEL_PATH}")
    print(f"Output model: {config.EMOTION_MODEL_OUTPUT_PATH}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max epochs: {config.MAX_EPOCHS}")
    print(f"Initial learning rate: {config.INITIAL_LEARNING_RATE}")
    print(f"Target learning rate: {config.TARGET_LEARNING_RATE}")
    print(f"Layer freezing: {'Enabled' if config.FREEZE_LAYERS else 'Disabled'}")
    print("=" * 80)
    
    # 1. Load FER+ dataset (original)
    print("\nLoading original FER+ dataset...")
    train_images, train_labels, test_images, test_labels = data_loader.load_ferplus_dataset()
    
    if train_images is None:
        print("Error loading original dataset. Exiting.")
        return
    
    # 2. First split original training data into train/validation sets
    print("\nSplitting original training data into train/validation sets...")
    orig_train_images, orig_val_images, orig_train_labels, orig_val_labels = train_test_split(
        train_images, 
        train_labels,
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=train_labels
    )
    
    print(f"Original training set: {len(orig_train_images)} samples")
    print(f"Validation set: {len(orig_val_images)} samples (from original data only)")
    print(f"Test set: {len(test_images)} samples")
    
    # 3. Load augmented FER+ dataset
    print("\nLoading augmented FER+ dataset...")
    augmented_images, augmented_labels = data_loader.load_augmented_fer_dataset()
    
    if augmented_images is None:
        print("Warning: Could not load augmented dataset. Proceeding with original data only.")
        # Use only original training data
        X_train = orig_train_images
        y_train = orig_train_labels
    else:
        # 4. Combine ONLY original training subset with augmented data
        print("\nCombining original training subset with augmented data...")
        X_train = np.concatenate([orig_train_images, augmented_images])
        y_train = np.concatenate([orig_train_labels, augmented_labels])
        
        # 5. Visualize examples of augmented data alongside originals
        visualize_augmentation_examples(
            augmented_images, 
            augmented_labels,
            original_images=train_images,
            original_labels=train_labels
        )
    
    # 6. Plot distribution of combined dataset
    visualization.plot_emotion_distribution(
        y_train, 
        "Training Data Emotion Distribution (with Augmented Data)",
        EMOTION_OUTPUT_DIR
    )
    
    # Also plot validation distribution
    visualization.plot_emotion_distribution(
        orig_val_labels, 
        "Validation Data Emotion Distribution (Original Data Only)",
        EMOTION_OUTPUT_DIR
    )
    
    # 7. Load pre-trained model
    model = load_pretrained_model()
    
    # 8. Fine-tune model with augmented data
    # Use original validation data only
    model, history, emotion_callback = fine_tune_model(
        model, 
        X_train, 
        y_train, 
        orig_val_images, 
        orig_val_labels
    )
    
    # 9. Plot training history
    visualization.plot_training_history(
        history, 
        EMOTION_OUTPUT_DIR
    )
    
    # 10. Plot emotion accuracy trends
    plot_emotion_accuracy_trends(emotion_callback)
    
    # 11. Evaluate on test set
    print("\nEvaluating model on test set...")
    test_results = metrics.print_evaluation_metrics(
        model, 
        test_images, 
        test_labels
    )
    
    # 12. Plot confusion matrix
    class_names = [config.FERPLUS_EMOTIONS[i] for i in range(config.NUM_CLASSES)]
    visualization.plot_confusion_matrix(
        test_results['confusion_matrix'],
        class_names,
        title='Emotion Recognition Confusion Matrix',
        save_path=os.path.join(EMOTION_OUTPUT_DIR, 'confusion_matrix.png')
    )
    
    # 13. Plot class accuracies
    visualization.plot_class_accuracies(
        test_results['class_accuracies'],
        title='Emotion Recognition Accuracy by Class',
        save_path=os.path.join(EMOTION_OUTPUT_DIR, 'class_accuracies.png')
    )
    
    # 14. Plot fairness metrics
    visualization.plot_fairness_bars(
        {'emotion_fairness': test_results['emotion_fairness']},
        title='Emotion Fairness Metric',
        save_path=os.path.join(EMOTION_OUTPUT_DIR, 'fairness_metrics.png')
    )
    
    # 15. Save test results
    with open(os.path.join(EMOTION_OUTPUT_DIR, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f)
    
    # Also save as CSV for easier analysis
    results_df = pd.DataFrame({
        'emotion': list(test_results['class_accuracies'].keys()),
        'accuracy': list(test_results['class_accuracies'].values())
    })
    results_df.to_csv(os.path.join(EMOTION_OUTPUT_DIR, 'test_results.csv'), index=False)
    
    print(f"\nFine-tuning and evaluation complete!")
    print(f"Results saved to {EMOTION_OUTPUT_DIR}")
    print(f"Fine-tuned model saved to {config.EMOTION_MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()