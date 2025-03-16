# bias_mitigation_approaches/data_augmentation/demographic_augmentation.py

"""
This script implements demographic-based bias mitigation for facial emotion recognition.
It loads a pre-trained FER+ model and fine-tunes it using augmented data from
the RAF-DB dataset to improve fairness across demographic groups.

Usage:
    python demographic_augmentation.py

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
DEMOGRAPHIC_OUTPUT_DIR = os.path.join(config.RESULTS_DIR, 'demographic_augmentation')
os.makedirs(DEMOGRAPHIC_OUTPUT_DIR, exist_ok=True)

#-----------------------------------------------------
# CUSTOM CALLBACKS
#-----------------------------------------------------

class DemographicFairnessCallback(tf.keras.callbacks.Callback):
    """Custom callback to track demographic fairness during training"""
    
    def __init__(self, validation_data):
        super(DemographicFairnessCallback, self).__init__()
        self.X_val, self.y_val, self.demographic_info = validation_data
        
        # Initialize tracking dictionaries
        self.gender_fairness_scores = []
        self.age_fairness_scores = []
        self.gender_accuracies = {gender: [] for gender in config.GENDER_LABELS.values()}
        self.age_accuracies = {age: [] for age in config.AGE_GROUPS.values()}
        
        # Extract demographic attributes for easier access
        self.gender_attributes = np.array([info['gender'] for info in self.demographic_info])
        self.age_attributes = np.array([info['age'] for info in self.demographic_info])
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get predictions (logits)
        y_pred_logits = self.model.predict(self.X_val, verbose=0)
        
        # Apply softmax to get probabilities
        y_pred_probs = metrics.softmax(y_pred_logits)
        
        # We need to convert FER+ predictions to RAF-DB emotion indices
        # First get the predicted FER+ classes
        ferplus_pred = np.argmax(y_pred_probs, axis=1)
        
        # Then map them to RAF-DB emotion indices
        rafdb_pred = np.array([config.FERPLUS_TO_RAFDB.get(p, 0) for p in ferplus_pred])
        
        # Get true RAF-DB labels
        y_true = self.y_val
        
        # Calculate demographic metrics
        gender_metrics = metrics.calculate_demographic_metrics(
            y_true, rafdb_pred, self.gender_attributes, 
            config.GENDER_LABELS, "Gender"
        )
        
        age_metrics = metrics.calculate_demographic_metrics(
            y_true, rafdb_pred, self.age_attributes, 
            config.AGE_GROUPS, "Age"
        )
        
        # Track metrics
        self.gender_fairness_scores.append(gender_metrics['fairness_score'])
        self.age_fairness_scores.append(age_metrics['fairness_score'])
        
        # Log metrics
        logs['gender_fairness'] = gender_metrics['fairness_score']
        logs['age_fairness'] = age_metrics['fairness_score']
        
        # Update accuracies without printing
        for group, acc in gender_metrics['accuracies'].items():
            self.gender_accuracies[group].append(acc)
        
        for group, acc in age_metrics['accuracies'].items():
            self.age_accuracies[group].append(acc)

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

def fine_tune_model(model, X_train, y_train, X_val, y_val, demographic_info_train, demographic_info_val):
    """Fine-tune the model with augmented data"""
    print("\nPreparing for fine-tuning...")
    
    # Convert RAF-DB labels to FER+ format for training
    # We'll need to create a mapping to go from RAF-DB to FER+
    rafdb_to_ferplus = {rafdb: ferplus for ferplus, rafdb in config.FERPLUS_TO_RAFDB.items() 
                        if rafdb is not None}
    
    # Convert labels (handle the case when mapping doesn't exist)
    y_train_ferplus = np.array([rafdb_to_ferplus.get(lbl, 0) for lbl in y_train])
    y_val_ferplus = np.array([rafdb_to_ferplus.get(lbl, 0) for lbl in y_val])
    
    # Convert to one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train_ferplus, config.NUM_CLASSES)
    y_val_onehot = tf.keras.utils.to_categorical(y_val_ferplus, config.NUM_CLASSES)

    # Compile the model with gradient clipping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.INITIAL_LEARNING_RATE,
            clipnorm=1.0  # Add gradient clipping for stability
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    # Initialize demographic fairness callback
    demographic_callback = DemographicFairnessCallback(
        validation_data=(X_val, y_val, demographic_info_val)
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
            filepath=config.DEMOGRAPHIC_MODEL_OUTPUT_PATH,
            monitor='val_loss',
            save_best_only=True,  # Only save when validation loss improves
            mode='min',          # Lower is better for loss
            verbose=1
        ),
        demographic_callback
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
    
    print(f"Fine-tuning complete. Best model saved to {config.DEMOGRAPHIC_MODEL_OUTPUT_PATH}")
    
    return model, history, demographic_callback

def plot_fairness_trends(demographic_callback, output_dir=DEMOGRAPHIC_OUTPUT_DIR):
    """Plot fairness trends during training"""
    epochs = range(1, len(demographic_callback.gender_fairness_scores) + 1)
    
    # Plot fairness scores
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, demographic_callback.gender_fairness_scores, 'b-', marker='o', label='Gender Fairness')
    plt.plot(epochs, demographic_callback.age_fairness_scores, 'r-', marker='s', label='Age Fairness')
    plt.title('Demographic Fairness Scores During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Fairness Score (min/max ratio)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'fairness_trends.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot gender accuracies
    plt.figure(figsize=(12, 6))
    for gender, accuracies in demographic_callback.gender_accuracies.items():
        plt.plot(epochs, accuracies, marker='o', linestyle='-', label=f'{gender}')
    plt.title('Gender Group Accuracies During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'gender_accuracy_trends.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot age accuracies
    plt.figure(figsize=(12, 6))
    for age, accuracies in demographic_callback.age_accuracies.items():
        plt.plot(epochs, accuracies, marker='o', linestyle='-', label=f'{age}')
    plt.title('Age Group Accuracies During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'age_accuracy_trends.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Save fairness data to CSV
    fairness_df = pd.DataFrame({
        'epoch': epochs,
        'gender_fairness': demographic_callback.gender_fairness_scores,
        'age_fairness': demographic_callback.age_fairness_scores
    })
    
    # Add demographic group accuracies
    for gender, accuracies in demographic_callback.gender_accuracies.items():
        fairness_df[f'gender_{gender.lower()}'] = accuracies
        
    for age, accuracies in demographic_callback.age_accuracies.items():
        age_key = age.replace(" ", "_").lower()
        fairness_df[f'age_{age_key}'] = accuracies
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'fairness_trends.csv')
    fairness_df.to_csv(csv_path, index=False)
    print(f"Fairness trends saved to {output_dir}")

#-----------------------------------------------------
# VISUALIZATION FUNCTIONS
#-----------------------------------------------------

def visualize_augmentation_examples(original_images, augmented_images, original_labels, augmented_labels, output_dir=DEMOGRAPHIC_OUTPUT_DIR):
    """Visualize examples of augmented samples alongside originals"""
    if augmented_images is not None and len(augmented_images) > 0:
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
                axes[displayed_count*2].set_title(f"Original {config.RAFDB_EMOTIONS.get(label, label)}")
                axes[displayed_count*2].axis('off')
                
                # Show augmented
                aug_idx = aug_indices[0]
                axes[displayed_count*2+1].imshow(augmented_images[aug_idx].squeeze(), cmap='gray')
                axes[displayed_count*2+1].set_title(f"Augmented {config.RAFDB_EMOTIONS.get(label, label)}")
                axes[displayed_count*2+1].axis('off')
                
                displayed_count += 1
        
        # Hide any unused axes
        for i in range(displayed_count*2, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'augmentation_examples.png'))
        plt.close()
        print(f"Augmentation examples visualization saved to {output_dir}")

#-----------------------------------------------------
# EVALUATION FUNCTIONS
#-----------------------------------------------------

def evaluate_demographic_fairness(model, X_test, y_test, demographic_info_test):
    """Evaluate demographic fairness on test set"""
    print("\nEvaluating demographic fairness on test set...")
    
    # Generate predictions
    y_pred_logits = model.predict(X_test)
    y_pred_probs = metrics.softmax(y_pred_logits)
    
    # Get predicted FER+ classes
    ferplus_pred = np.argmax(y_pred_probs, axis=1)
    
    # Map to RAF-DB emotion indices
    rafdb_pred = np.array([config.FERPLUS_TO_RAFDB.get(p, 0) for p in ferplus_pred])
    
    # Extract demographic attributes
    gender_attributes = np.array([info['gender'] for info in demographic_info_test])
    age_attributes = np.array([info['age'] for info in demographic_info_test])
    
    # Calculate demographic metrics
    gender_metrics = metrics.calculate_demographic_metrics(
        y_test, rafdb_pred, gender_attributes, 
        config.GENDER_LABELS, "Gender"
    )
    
    age_metrics = metrics.calculate_demographic_metrics(
        y_test, rafdb_pred, age_attributes, 
        config.AGE_GROUPS, "Age"
    )
    
    # Calculate emotion accuracies
    emotion_accuracies = {}
    for raf_label, emotion_name in config.RAFDB_EMOTIONS.items():
        mask = y_test == raf_label
        if np.any(mask):
            acc = np.mean(rafdb_pred[mask] == raf_label)
            emotion_accuracies[emotion_name] = acc
    
    # Print fairness scores
    print(f"Gender Fairness Score: {gender_metrics['fairness_score'] * 100:.2f}%")
    print("Gender Group Accuracies:")
    for group, acc in gender_metrics['accuracies'].items():
        print(f"  {group}: {acc * 100:.2f}%")
    
    print(f"Age Fairness Score: {age_metrics['fairness_score'] * 100:.2f}%")
    print("Age Group Accuracies:")
    for group, acc in age_metrics['accuracies'].items():
        print(f"  {group}: {acc * 100:.2f}%")
    
    # Print emotion accuracies
    print("\nEmotion Accuracies on RAF-DB:")
    for emotion, acc in emotion_accuracies.items():
        print(f"  {emotion}: {acc * 100:.2f}%")
    
    # Return results
    return {
        'gender_metrics': gender_metrics,
        'age_metrics': age_metrics,
        'emotion_accuracies': emotion_accuracies
    }

#-----------------------------------------------------
# MAIN EXECUTION FLOW
#-----------------------------------------------------

def main():
    # Print configuration information
    print("=" * 80)
    print("DEMOGRAPHIC-BASED AUGMENTATION FINE-TUNING")
    print("=" * 80)
    print(f"Base model: {config.BASE_MODEL_PATH}")
    print(f"Output model: {config.DEMOGRAPHIC_MODEL_OUTPUT_PATH}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max epochs: {config.MAX_EPOCHS}")
    print(f"Initial learning rate: {config.INITIAL_LEARNING_RATE}")
    print(f"Target learning rate: {config.TARGET_LEARNING_RATE}")
    print(f"Layer freezing: {'Enabled' if config.FREEZE_LAYERS else 'Disabled'}")
    print("=" * 80)
    
    # 1. Load RAF-DB dataset (original)
    print("\nLoading original RAF-DB dataset...")
    train_images, train_labels, train_demographic_info = data_loader.load_rafdb_dataset('train')
    test_images, test_labels, test_demographic_info = data_loader.load_rafdb_dataset('test')
    
    if train_images is None or test_images is None:
        print("Error loading RAF-DB dataset. Exiting.")
        return
    
    # 2. First split original training data into train/validation sets
    print("\nSplitting original training data into train/validation sets...")
    # Need to keep track of indices to properly split demographic info
    train_indices = np.arange(len(train_images))
    
    orig_train_images, orig_val_images, orig_train_labels, orig_val_labels, orig_train_indices, orig_val_indices = train_test_split(
        train_images, 
        train_labels,
        train_indices,
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=train_labels
    )
    
    # Split demographic info based on indices
    orig_train_demo_info = [train_demographic_info[i] for i in orig_train_indices]
    orig_val_demo_info = [train_demographic_info[i] for i in orig_val_indices]
    
    print(f"Original training set: {len(orig_train_images)} samples")
    print(f"Validation set: {len(orig_val_images)} samples (from original data only)")
    print(f"Test set: {len(test_images)} samples")
    
    # 3. Load augmented RAF-DB dataset
    print("\nLoading augmented RAF-DB dataset...")
    augmented_images, augmented_labels, augmented_demographic_info = data_loader.load_augmented_rafdb_dataset()
    
    if augmented_images is None:
        print("Warning: Could not load augmented dataset. Proceeding with original data only.")
        # Use only original training data
        X_train = orig_train_images
        y_train = orig_train_labels
        train_demo_info = orig_train_demo_info
    else:
        # 4. Combine ONLY original training subset with augmented data
        print("\nCombining original training subset with augmented data...")
        X_train = np.concatenate([orig_train_images, augmented_images])
        y_train = np.concatenate([orig_train_labels, augmented_labels])
        
        # Combine demographic information
        train_demo_info = orig_train_demo_info.copy()
        if augmented_demographic_info:
            train_demo_info.extend(augmented_demographic_info)
            
        # 5. Visualize examples of augmented data alongside originals
        visualize_augmentation_examples(
            train_images,
            augmented_images,
            train_labels,
            augmented_labels
        )
    
    # 6. Plot distributions
    # Emotion distribution of training data
    emotion_labels = [config.RAFDB_EMOTIONS.get(label, str(label)) for label in y_train]
    visualization.plot_emotion_distribution(
        y_train,
        "Training Data Emotion Distribution (with Augmented Data)",
        DEMOGRAPHIC_OUTPUT_DIR
    )
    
    # Emotion distribution of validation data
    val_emotion_labels = [config.RAFDB_EMOTIONS.get(label, str(label)) for label in orig_val_labels]
    visualization.plot_emotion_distribution(
        orig_val_labels,
        "Validation Data Emotion Distribution (Original Data Only)",
        DEMOGRAPHIC_OUTPUT_DIR
    )
    
    # Demographic distribution of training data
    visualization.plot_demographic_distribution(
        train_demo_info,
        DEMOGRAPHIC_OUTPUT_DIR
    )
    
    # 7. Load pre-trained model
    model = load_pretrained_model()
    
    # 8. Fine-tune model with augmented data
    # Use original validation data only
    model, history, demographic_callback = fine_tune_model(
        model, 
        X_train, 
        y_train, 
        orig_val_images, 
        orig_val_labels, 
        train_demo_info, 
        orig_val_demo_info
    )
    
    # 9. Plot training history
    visualization.plot_training_history(
        history, 
        DEMOGRAPHIC_OUTPUT_DIR
    )
    
    # 10. Plot fairness trends
    plot_fairness_trends(demographic_callback)
    
    # 11. Evaluate on test set
    print("\nEvaluating model on test set...")
    test_results = evaluate_demographic_fairness(
        model, 
        test_images, 
        test_labels, 
        test_demographic_info
    )
    
    # 12. Plot class accuracies
    visualization.plot_class_accuracies(
        test_results['emotion_accuracies'],
        title='Emotion Recognition Accuracy by Class',
        save_path=os.path.join(DEMOGRAPHIC_OUTPUT_DIR, 'emotion_accuracies.png')
    )
    
    # 13. Plot fairness metrics
    fairness_metrics = {
        'gender_fairness': test_results['gender_metrics']['fairness_score'],
        'age_fairness': test_results['age_metrics']['fairness_score']
    }
    
    visualization.plot_fairness_bars(
        fairness_metrics,
        title='Demographic Fairness Metrics',
        save_path=os.path.join(DEMOGRAPHIC_OUTPUT_DIR, 'fairness_metrics.png')
    )
    
    # 14. Save detailed demographic results
    with open(os.path.join(DEMOGRAPHIC_OUTPUT_DIR, 'demographic_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f)
    
    # Also save as CSV for easier analysis
    # Gender results
    gender_df = pd.DataFrame({
        'gender': list(test_results['gender_metrics']['accuracies'].keys()),
        'accuracy': list(test_results['gender_metrics']['accuracies'].values())
    })
    gender_df.to_csv(os.path.join(DEMOGRAPHIC_OUTPUT_DIR, 'gender_results.csv'), index=False)
    
    # Age results
    age_df = pd.DataFrame({
        'age_group': list(test_results['age_metrics']['accuracies'].keys()),
        'accuracy': list(test_results['age_metrics']['accuracies'].values())
    })
    age_df.to_csv(os.path.join(DEMOGRAPHIC_OUTPUT_DIR, 'age_results.csv'), index=False)
    
    # Emotion results
    emotion_df = pd.DataFrame({
        'emotion': list(test_results['emotion_accuracies'].keys()),
        'accuracy': list(test_results['emotion_accuracies'].values())
    })
    emotion_df.to_csv(os.path.join(DEMOGRAPHIC_OUTPUT_DIR, 'emotion_results.csv'), index=False)
    
    print(f"\nFine-tuning and evaluation complete!")
    print(f"Results saved to {DEMOGRAPHIC_OUTPUT_DIR}")
    print(f"Fine-tuned model saved to {config.DEMOGRAPHIC_MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()