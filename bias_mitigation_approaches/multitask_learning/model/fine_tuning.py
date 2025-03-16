# bias_mitigation_approaches/multitask_learning/model/fine_tuning.py

import tensorflow as tf
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utils.config import *

#-----------------------------------------------------
# CUSTOM CALLBACKS
#-----------------------------------------------------

class WarmupScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler with warmup phase"""
    
    def __init__(self, warmup_epochs, initial_lr, target_lr):
        super(WarmupScheduler, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            print(f"\nEpoch {epoch+1}: Learning rate set to {lr:.2e} (warmup phase)")

class DemographicFairnessCallback(tf.keras.callbacks.Callback):
    """Custom callback to track demographic fairness during training"""
    
    def __init__(self, validation_data, validation_demographic_info):
        super(DemographicFairnessCallback, self).__init__()
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]  # List of [emotion_labels, gender_labels, age_labels]
        
        # Extract demographic attributes for each sample
        self.gender_attributes = np.argmax(self.y_val[1], axis=1)
        self.age_attributes = np.argmax(self.y_val[2], axis=1)
        
        # Initialize tracking dictionaries
        self.gender_fairness_scores = []
        self.age_fairness_scores = []
        self.emotion_fairness_scores = []
        
        # Track group accuracies
        self.gender_group_accuracies = {gender: [] for gender in GENDER_LABELS.values()}
        self.age_group_accuracies = {age: [] for age in AGE_GROUPS.values()}
        
        # Ensure all emotions have tracking arrays, even if not present in validation set
        self.emotion_accuracies = {emotion: [] for emotion in FERPLUS_EMOTIONS.values()}
        
        # Store demographic information
        self.validation_demographic_info = validation_demographic_info
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get predictions
        predictions = self.model.predict(self.X_val, verbose=0)
        
        # Extract true labels
        y_true_emotion = np.argmax(self.y_val[0], axis=1)
        
        # Extract predicted labels
        y_pred_emotion = np.argmax(predictions[0], axis=1)
        
        # Calculate emotion accuracies
        emotion_accuracies = {}
        
        # Initialize with zeros for all emotions
        for emotion_idx in range(NUM_EMOTION_CLASSES):
            emotion_name = FERPLUS_EMOTIONS[emotion_idx]
            emotion_accuracies[emotion_name] = 0.0
        
        # Update with actual accuracies where data exists
        for emotion_idx in range(NUM_EMOTION_CLASSES):
            mask = y_true_emotion == emotion_idx
            if np.any(mask):
                accuracy = np.mean(y_pred_emotion[mask] == emotion_idx)
                emotion_name = FERPLUS_EMOTIONS[emotion_idx]
                emotion_accuracies[emotion_name] = accuracy
        
        # Update tracking arrays for all emotions
        for emotion_name in FERPLUS_EMOTIONS.values():
            self.emotion_accuracies[emotion_name].append(emotion_accuracies.get(emotion_name, 0.0))
        
        # Calculate emotion fairness
        if emotion_accuracies:
            min_emotion_acc = min(emotion_accuracies.values())
            max_emotion_acc = max(emotion_accuracies.values())
            emotion_fairness = min_emotion_acc / max_emotion_acc if max_emotion_acc > 0 else 0
        else:
            emotion_fairness = 0
        self.emotion_fairness_scores.append(emotion_fairness)
        
        # Calculate gender group accuracies
        gender_group_accuracies = {}
        for gender_idx, gender_name in GENDER_LABELS.items():
            mask = self.gender_attributes == gender_idx
            if np.any(mask):
                accuracy = np.mean(y_pred_emotion[mask] == y_true_emotion[mask])
                gender_group_accuracies[gender_name] = accuracy
                self.gender_group_accuracies[gender_name].append(accuracy)
            else:
                # If no samples for this gender, use previous value or 0
                previous_value = self.gender_group_accuracies[gender_name][-1] if self.gender_group_accuracies[gender_name] else 0
                self.gender_group_accuracies[gender_name].append(previous_value)
        
        # Calculate gender fairness
        if gender_group_accuracies:
            min_gender_acc = min(gender_group_accuracies.values())
            max_gender_acc = max(gender_group_accuracies.values())
            gender_fairness = min_gender_acc / max_gender_acc if max_gender_acc > 0 else 0
        else:
            gender_fairness = 0
        self.gender_fairness_scores.append(gender_fairness)
        
        # Calculate age group accuracies
        age_group_accuracies = {}
        for age_idx, age_name in AGE_GROUPS.items():
            # Adjust age index (from 1-5 to 0-4)
            adjusted_idx = age_idx - 1
            mask = self.age_attributes == adjusted_idx
            if np.any(mask):
                accuracy = np.mean(y_pred_emotion[mask] == y_true_emotion[mask])
                age_group_accuracies[age_name] = accuracy
                self.age_group_accuracies[age_name].append(accuracy)
            else:
                # If no samples for this age group, use previous value or 0
                previous_value = self.age_group_accuracies[age_name][-1] if self.age_group_accuracies[age_name] else 0
                self.age_group_accuracies[age_name].append(previous_value)
        
        # Calculate age fairness
        if age_group_accuracies:
            min_age_acc = min(age_group_accuracies.values())
            max_age_acc = max(age_group_accuracies.values())
            age_fairness = min_age_acc / max_age_acc if max_age_acc > 0 else 0
        else:
            age_fairness = 0
        self.age_fairness_scores.append(age_fairness)
        
        # Log fairness metrics
        logs['emotion_fairness'] = emotion_fairness
        logs['gender_fairness'] = gender_fairness
        logs['age_fairness'] = age_fairness
        
        # Print fairness metrics
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Print every 5 epochs or first epoch
            print(f"\nFairness metrics at epoch {epoch+1}:")
            print(f"  Emotion Fairness: {emotion_fairness:.4f}")
            print(f"  Gender Fairness: {gender_fairness:.4f}")
            print(f"  Age Fairness: {age_fairness:.4f}")

#-----------------------------------------------------
# TRAINING FUNCTIONS
#-----------------------------------------------------

def train_heads_only(model, feature_extractor, train_data, val_data, val_demographic_info, epochs=PHASE1_EPOCHS):
    """
    First training phase: Train only the new task heads with frozen feature extractor
    
    Args:
        model: Multi-task model
        feature_extractor: Feature extraction layers
        train_data: Training data tuple (x, [y_emotion, y_gender, y_age])
        val_data: Validation data tuple
        val_demographic_info: Validation demographic information
        epochs: Number of epochs for this phase
        
    Returns:
        Training history and fairness callback
    """
    print("\nPhase 1: Training only the task-specific heads...")
    
    # Freeze feature extractor
    for layer in feature_extractor.layers:
        layer.trainable = False
    
    # Initialize demographic fairness callback
    fairness_callback = DemographicFairnessCallback(val_data, val_demographic_info)
    
    # Initialize warmup scheduler
    warmup_scheduler = WarmupScheduler(
        warmup_epochs=WARMUP_EPOCHS_PHASE1,
        initial_lr=INITIAL_LR_PHASE1 / 10,  # Start with lower learning rate
        target_lr=INITIAL_LR_PHASE1
    )
    
    # Compile model with higher learning rate for new task heads and gradient clipping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LR_PHASE1,
            clipnorm=GRADIENT_CLIP_NORM  # Add gradient clipping
        ),
        loss={
            'emotion_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            'gender_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            'age_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            'emotion_output': EMOTION_LOSS_WEIGHT,
            'gender_output': GENDER_LOSS_WEIGHT,
            'age_output': AGE_LOSS_WEIGHT
        },
        metrics=['accuracy']
    )
    
    # Train only the heads
    history = model.fit(
        train_data[0],
        train_data[1],
        validation_data=val_data,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[warmup_scheduler, fairness_callback],
        verbose=2  # Show one line per epoch
    )
    
    return history, fairness_callback

def fine_tune_all_layers(model, feature_extractor, train_data, val_data, val_demographic_info, callbacks=None):
    """
    Second training phase: Fine-tune all layers with lower learning rate
    
    Args:
        model: Multi-task model with trained heads
        feature_extractor: Feature extraction layers
        train_data: Training data tuple (x, [y_emotion, y_gender, y_age])
        val_data: Validation data tuple
        val_demographic_info: Validation demographic information
        callbacks: List of Keras callbacks
        
    Returns:
        Training history and fairness callback
    """
    print("\nPhase 2: Fine-tuning all layers...")
    
    # Unfreeze feature extractor
    for layer in feature_extractor.layers:
        layer.trainable = True
    
    # Initialize demographic fairness callback
    fairness_callback = DemographicFairnessCallback(val_data, val_demographic_info)
    
    # Initialize warmup scheduler
    warmup_scheduler = WarmupScheduler(
        warmup_epochs=WARMUP_EPOCHS_PHASE2,
        initial_lr=INITIAL_LR_PHASE2 / 5,  # Start with lower learning rate
        target_lr=INITIAL_LR_PHASE2
    )
    
    # Add fairness callback and warmup scheduler to other callbacks
    if callbacks is None:
        callbacks = []
    callbacks.extend([fairness_callback, warmup_scheduler])
    
    # Recompile model with lower learning rate and gradient clipping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LR_PHASE2,
            clipnorm=GRADIENT_CLIP_NORM  # Add gradient clipping
        ),
        loss={
            'emotion_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            'gender_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            'age_output': tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            'emotion_output': EMOTION_LOSS_WEIGHT,
            'gender_output': GENDER_LOSS_WEIGHT,
            'age_output': AGE_LOSS_WEIGHT
        },
        metrics=['accuracy']
    )
    
    # Fine-tune all layers
    history = model.fit(
        train_data[0],
        train_data[1],
        validation_data=val_data,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2  # Show one line per epoch
    )
    
    return history, fairness_callback

def create_fairness_aware_callbacks(val_emotion_accuracy_metric='val_emotion_output_accuracy', 
                                   model_save_path=MULTITASK_MODEL_OUTPUT_PATH):
    """Create callbacks for training with fairness awareness"""
    
    callbacks = [
        # Early stopping based on validation emotion accuracy
        tf.keras.callbacks.EarlyStopping(
            monitor=val_emotion_accuracy_metric,
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        # Model checkpoint to save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor=val_emotion_accuracy_metric,
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            mode='min',
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(TRAINING_HISTORY_DIR, 'logs'),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks

def progressive_fine_tuning(model, feature_extractor, train_data, val_data, 
                           val_demographic_info=None,
                           model_save_path=MULTITASK_MODEL_OUTPUT_PATH):
    """
    Perform progressive fine-tuning: First train only the heads, then fine-tune all layers
    
    Args:
        model: Multi-task model
        feature_extractor: Feature extraction layers
        train_data: Training data tuple (x, [y_emotion, y_gender, y_age])
        val_data: Validation data tuple
        val_demographic_info: Validation demographic information
        model_save_path: Path to save the best model
        
    Returns:
        Combined training history and fairness metrics
    """
    # If demographic info not provided, create a placeholder (empty list)
    if val_demographic_info is None:
        val_demographic_info = []
    
    # Phase 1: Train only the heads
    phase1_history, phase1_fairness = train_heads_only(
        model, 
        feature_extractor, 
        train_data, 
        val_data,
        val_demographic_info,
        epochs=PHASE1_EPOCHS
    )
    
    # Set up callbacks for phase 2
    callbacks = create_fairness_aware_callbacks(
        val_emotion_accuracy_metric='val_emotion_output_accuracy',
        model_save_path=model_save_path
    )
    
    # Phase 2: Fine-tune all layers
    phase2_history, phase2_fairness = fine_tune_all_layers(
        model, 
        feature_extractor, 
        train_data, 
        val_data,
        val_demographic_info,
        callbacks=callbacks
    )
    
    # Combine histories for visualization
    combined_history = {}
    for key in phase2_history.history:
        if key in phase1_history.history:
            combined_history[key] = phase1_history.history[key] + phase2_history.history[key]
        else:
            # If key only exists in phase 2, pad with zeros for phase 1
            combined_history[key] = [0] * len(phase1_history.history['loss']) + phase2_history.history[key]
    
    # Create a history-like object
    history_obj = type('obj', (object,), {'history': combined_history})
    
    # Combine fairness metrics
    fairness_metrics = {
        'gender_fairness_scores': phase1_fairness.gender_fairness_scores + phase2_fairness.gender_fairness_scores,
        'age_fairness_scores': phase1_fairness.age_fairness_scores + phase2_fairness.age_fairness_scores,
        'emotion_fairness_scores': phase1_fairness.emotion_fairness_scores + phase2_fairness.emotion_fairness_scores,
        'gender_group_accuracies': {},
        'age_group_accuracies': {},
        'emotion_accuracies': {}
    }
    
    # Combine group accuracies
    for gender in GENDER_LABELS.values():
        fairness_metrics['gender_group_accuracies'][gender] = (
            phase1_fairness.gender_group_accuracies[gender] + 
            phase2_fairness.gender_group_accuracies[gender]
        )
    
    for age in AGE_GROUPS.values():
        fairness_metrics['age_group_accuracies'][age] = (
            phase1_fairness.age_group_accuracies[age] + 
            phase2_fairness.age_group_accuracies[age]
        )
    
    for emotion in FERPLUS_EMOTIONS.values():
        fairness_metrics['emotion_accuracies'][emotion] = (
            phase1_fairness.emotion_accuracies[emotion] + 
            phase2_fairness.emotion_accuracies[emotion]
        )
    
    return history_obj, fairness_metrics