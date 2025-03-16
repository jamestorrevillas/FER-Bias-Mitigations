# bias_mitigation_approaches/data_augmentation/utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from . import config

def plot_emotion_distribution(labels, title="Emotion Distribution", save_dir=None):
    """
    Plot emotion class distribution
    
    Args:
        labels: Array of emotion labels
        title: Plot title
        save_dir: Optional directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Count frequencies
    unique_labels, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # Map indices to emotion names
    emotions = [config.FERPLUS_EMOTIONS.get(label, str(label)) for label in unique_labels]
    
    # Calculate percentages
    percentages = (counts / total) * 100
    
    # Create bars
    bars = plt.bar(emotions, counts)
    
    # Add count and percentage labels
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom'
        )
    
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel('Emotion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'emotion_distribution.png'), dpi=300)
    
    plt.close()

def plot_demographic_distribution(demographic_info, save_dir=None):
    """
    Plot demographic distributions
    
    Args:
        demographic_info: List of dictionaries with demographic information
        save_dir: Optional directory to save the plots
    """
    # Convert to pandas DataFrame for easier analysis
    demo_df = pd.DataFrame(demographic_info)
    
    # Plot gender distribution
    plt.figure(figsize=(10, 5))
    
    gender_counts = demo_df['gender'].value_counts().sort_index()
    gender_labels = [config.GENDER_LABELS.get(idx, str(idx)) for idx in gender_counts.index]
    
    plt.subplot(1, 2, 1)
    gender_bars = plt.bar(gender_labels, gender_counts.values)
    plt.title('Gender Distribution')
    plt.ylabel('Count')
    
    for bar in gender_bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f'{int(bar.get_height())}',
            ha='center', va='bottom'
        )
    
    # Plot age distribution
    age_counts = demo_df['age'].value_counts().sort_index()
    age_labels = [config.AGE_GROUPS.get(idx, str(idx)) for idx in age_counts.index]
    
    plt.subplot(1, 2, 2)
    age_bars = plt.bar(age_labels, age_counts.values)
    plt.title('Age Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    for bar in age_bars:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f'{int(bar.get_height())}',
            ha='center', va='bottom'
        )
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'demographic_distribution.png'), dpi=300)
    
    plt.close()

def plot_training_history(history, output_dir=None):
    """
    Plot and save training metrics
    
    Args:
        history: Keras history object from model training
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = config.TRAINING_HISTORY_DIR
    
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history.history['accuracy']) + 1)
    plt.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()
    
    print(f"Training history plot saved to {output_dir}")
    
    # Save training history to CSV
    history_df = pd.DataFrame({
        'epoch': epochs,
        'train_accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })
    
    # Add demographic fairness metrics if they exist in history
    for key in history.history:
        if key.startswith('gender_') or key.startswith('age_'):
            history_df[key] = history.history[key]
    
    history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    print(f"Training history CSV saved to {output_dir}")

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        title: Plot title
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    plt.close()

def plot_class_accuracies(class_accuracies, title='Class Accuracies', save_path=None):
    """
    Plot class-wise accuracies
    
    Args:
        class_accuracies: Dictionary mapping class names to accuracies
        title: Plot title
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 6))
    
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    bars = plt.bar(classes, accuracies)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01,
            f'{height:.4f}',
            ha='center', 
            va='bottom'
        )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    plt.close()

def plot_fairness_bars(fairness_metrics, title='Fairness Metrics', save_path=None):
    """
    Plot fairness metrics
    
    Args:
        fairness_metrics: Dictionary of fairness metrics
        title: Plot title
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(8, 6))
    
    metrics = []
    values = []
    
    # Extract metrics to plot
    if 'emotion_fairness' in fairness_metrics:
        metrics.append('Emotion Fairness')
        values.append(fairness_metrics['emotion_fairness'])
    
    if 'gender_fairness' in fairness_metrics:
        metrics.append('Gender Fairness')
        values.append(fairness_metrics['gender_fairness'])
    
    if 'age_fairness' in fairness_metrics:
        metrics.append('Age Fairness')
        values.append(fairness_metrics['age_fairness'])
    
    # Create bars
    bars = plt.bar(metrics, values)
    plt.title(title)
    plt.ylabel('Fairness Score (min/max ratio)')
    plt.ylim(0, 1.0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01,
            f'{height:.4f}',
            ha='center', 
            va='bottom'
        )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    plt.close()