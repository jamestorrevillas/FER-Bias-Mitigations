# bias_mitigation_approaches/dynamic_weighting/utils/visualization.py

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

def plot_fairness_trends(fairness_history, save_path=None):
    """
    Plot fairness metric trends over training
    
    Args:
        fairness_history: Dictionary with fairness metric lists
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    iterations = range(len(fairness_history['gender_fairness']))
    
    # Plot gender fairness
    plt.plot(iterations, fairness_history['gender_fairness'], 'b-', 
             marker='o', label='Gender Fairness')
    
    # Plot age fairness if available
    if 'age_fairness' in fairness_history:
        plt.plot(iterations, fairness_history['age_fairness'], 'r-',
                marker='s', label='Age Fairness')
    
    # Plot emotion fairness if available
    if 'emotion_fairness' in fairness_history:
        plt.plot(iterations, fairness_history['emotion_fairness'], 'g-',
                marker='^', label='Emotion Fairness')
    
    plt.title('Fairness Metrics During Training')
    plt.xlabel('Feedback Iteration')
    plt.ylabel('Fairness Score (min/max ratio)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    plt.close()

def plot_weight_evolution(weight_history, save_path=None):
    """
    Plot evolution of sample weights over training
    
    Args:
        weight_history: List of weight distribution statistics
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    iterations = range(len(weight_history))
    
    # Extract statistics
    means = [stats['mean'] for stats in weight_history]
    medians = [stats['median'] for stats in weight_history]
    maxs = [stats['max'] for stats in weight_history]
    mins = [stats['min'] for stats in weight_history]
    stds = [stats['std'] for stats in weight_history]
    
    # Plot mean and median
    plt.plot(iterations, means, 'b-', marker='o', label='Mean Weight')
    plt.plot(iterations, medians, 'g--', marker='s', label='Median Weight')
    
    # Plot min and max with shaded region for range
    plt.fill_between(iterations, mins, maxs, alpha=0.2, color='gray', label='Min-Max Range')
    
    # Plot standard deviation
    plt.plot(iterations, stds, 'r-.', marker='^', label='Standard Deviation')
    
    plt.title('Sample Weight Evolution During Training')
    plt.xlabel('Feedback Iteration')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    plt.close()

def plot_group_performance(metrics_history, group_type='gender', save_path=None):
    """
    Plot performance trends for demographic groups
    
    Args:
        metrics_history: List of metric dictionaries
        group_type: 'gender' or 'age'
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    iterations = range(len(metrics_history))
    
    # Extract group data based on type
    if group_type == 'gender':
        # Track all gender groups
        group_data = {}
        
        for i, metrics in enumerate(metrics_history):
            gender_metrics = metrics['gender_metrics']
            for group, acc in gender_metrics['accuracies'].items():
                if group not in group_data:
                    group_data[group] = [None] * len(metrics_history)
                group_data[group][i] = acc
    
    elif group_type == 'age':
        # Track all age groups
        group_data = {}
        
        for i, metrics in enumerate(metrics_history):
            age_metrics = metrics['age_metrics']
            for group, acc in age_metrics['accuracies'].items():
                if group not in group_data:
                    group_data[group] = [None] * len(metrics_history)
                group_data[group][i] = acc
    
    # Plot each group
    for group, accuracies in group_data.items():
        # Filter out None values
        valid_indices = [i for i, acc in enumerate(accuracies) if acc is not None]
        valid_accuracies = [accuracies[i] for i in valid_indices]
        
        if valid_indices:
            plt.plot(valid_indices, valid_accuracies, marker='o', label=group)
    
    plt.title(f'{group_type.capitalize()} Group Performance During Training')
    plt.xlabel('Feedback Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    plt.close()

def plot_emotion_accuracies(emotion_accuracies, title='Emotion Accuracies', save_path=None):
    """
    Plot accuracies for each emotion category
    
    Args:
        emotion_accuracies: Dictionary of emotion accuracies
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    emotions = list(emotion_accuracies.keys())
    accuracies = list(emotion_accuracies.values())
    
    bars = plt.bar(emotions, accuracies)
    plt.title(title)
    plt.xlabel('Emotion')
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

def plot_intersectional_performance(fairness_metrics, save_path=None):
    """
    Plot heatmap of demographic-emotion intersectional performance
    
    Args:
        fairness_metrics: Output from evaluate_model_fairness
        save_path: Path to save the heatmap
    """
    # Extract intersectional data
    gender_metrics = fairness_metrics['gender_metrics']
    
    # Create a DataFrame for gender-emotion intersection
    gender_emotion_data = {}
    
    for gender, emotions in gender_metrics['emotion_by_demo'].items():
        gender_emotion_data[gender] = emotions
    
    gender_df = pd.DataFrame(gender_emotion_data)
    
    # Create gender-emotion heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(gender_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.3f')
    plt.title('Gender-Emotion Intersection Performance')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    plt.close()
    
    # Create age-emotion heatmap if available
    if 'age_metrics' in fairness_metrics:
        age_metrics = fairness_metrics['age_metrics']
        
        # Create a DataFrame for age-emotion intersection
        age_emotion_data = {}
        
        for age, emotions in age_metrics['emotion_by_demo'].items():
            age_emotion_data[age] = emotions
        
        age_df = pd.DataFrame(age_emotion_data)
        
        # Create age-emotion heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(age_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1, fmt='.3f')
        plt.title('Age-Emotion Intersection Performance')
        plt.tight_layout()
        
        if save_path:
            age_path = os.path.join(os.path.dirname(save_path), 'age_emotion_heatmap.png')
            plt.savefig(age_path, dpi=300)
        
        plt.close()