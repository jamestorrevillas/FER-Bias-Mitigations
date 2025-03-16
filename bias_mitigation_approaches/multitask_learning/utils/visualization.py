# bias_mitigation_approaches/multitask_learning/utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import sys

# Append parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bias_mitigation_approaches.multitask_learning.utils.config import *

def plot_training_history(history, output_dir=None):
    """
    Plot training history metrics
    
    Args:
        history: Keras history object
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = TRAINING_HISTORY_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    metrics = history.history
    epochs = range(1, len(metrics['loss']) + 1)
    
    # Plot task losses
    plt.figure(figsize=(15, 5))
    
    # Plot overall loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss')
    plt.title('Overall Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot emotion task
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics['emotion_output_loss'], 'b-', label='Training')
    plt.plot(epochs, metrics['val_emotion_output_loss'], 'r-', label='Validation')
    plt.title('Emotion Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics['emotion_output_accuracy'], 'b-', label='Emotion')
    plt.plot(epochs, metrics['gender_output_accuracy'], 'g-', label='Gender')
    plt.plot(epochs, metrics['age_output_accuracy'], 'r-', label='Age')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()
    
    # Plot validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['val_emotion_output_accuracy'], 'b-', label='Emotion')
    plt.plot(epochs, metrics['val_gender_output_accuracy'], 'g-', label='Gender')
    plt.plot(epochs, metrics['val_age_output_accuracy'], 'r-', label='Age')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_accuracy.png'), dpi=300)
    plt.close()
    
    # If fairness metrics are available, plot them
    fairness_metrics = ['emotion_fairness', 'gender_fairness', 'age_fairness']
    if all(metric in metrics for metric in fairness_metrics):
        plt.figure(figsize=(10, 5))
        for metric in fairness_metrics:
            plt.plot(epochs, metrics[metric], label=metric.replace('_', ' ').title())
        plt.title('Fairness Metrics During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Fairness Score')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fairness_metrics.png'), dpi=300)
        plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': epochs,
        'loss': metrics['loss'],
        'val_loss': metrics['val_loss'],
        'emotion_loss': metrics['emotion_output_loss'],
        'val_emotion_loss': metrics['val_emotion_output_loss'],
        'gender_loss': metrics['gender_output_loss'],
        'val_gender_loss': metrics['val_gender_output_loss'],
        'age_loss': metrics['age_output_loss'],
        'val_age_loss': metrics['val_age_output_loss'],
        'emotion_acc': metrics['emotion_output_accuracy'],
        'val_emotion_acc': metrics['val_emotion_output_accuracy'],
        'gender_acc': metrics['gender_output_accuracy'],
        'val_gender_acc': metrics['val_gender_output_accuracy'],
        'age_acc': metrics['age_output_accuracy'],
        'val_age_acc': metrics['val_age_output_accuracy']
    })
    
    # Add fairness metrics if available
    for metric in fairness_metrics:
        if metric in metrics:
            metrics_df[metric] = metrics[metric]
    
    metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)
    print(f"Training history saved to {output_dir}")

def plot_fairness_trends(fairness_metrics, output_dir=None):
    """
    Plot fairness trends during training
    
    Args:
        fairness_metrics: Dictionary with fairness metrics by epoch
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = FAIRNESS_TRENDS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot fairness scores
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(fairness_metrics['gender_fairness_scores']) + 1)
    
    plt.plot(epochs, fairness_metrics['gender_fairness_scores'], 'b-', marker='o', markersize=3, 
             label='Gender Fairness')
    plt.plot(epochs, fairness_metrics['age_fairness_scores'], 'r-', marker='s', markersize=3, 
             label='Age Fairness')
    plt.plot(epochs, fairness_metrics['emotion_fairness_scores'], 'g-', marker='^', markersize=3, 
             label='Emotion Fairness')
    
    plt.title('Demographic Fairness Scores During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Fairness Score (min/max ratio)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    # Add phase separator
    if len(epochs) > PHASE1_EPOCHS:
        plt.axvline(x=PHASE1_EPOCHS, color='k', linestyle='--', alpha=0.5)
        plt.text(PHASE1_EPOCHS, 0.5, "Phase 1 | Phase 2", 
                 fontsize=10, horizontalalignment='center', verticalalignment='center',
                 rotation=90, transform=plt.gca().get_xaxis_transform())
    
    save_path = os.path.join(output_dir, 'fairness_trends.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot gender group accuracies
    plt.figure(figsize=(12, 6))
    for gender, accuracies in fairness_metrics['gender_group_accuracies'].items():
        plt.plot(epochs, accuracies, marker='o', markersize=3, linestyle='-', label=f'{gender}')
    
    plt.title('Gender Group Accuracies During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    
    # Add phase separator
    if len(epochs) > PHASE1_EPOCHS:
        plt.axvline(x=PHASE1_EPOCHS, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'gender_accuracy_trends.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot age group accuracies
    plt.figure(figsize=(12, 6))
    for age, accuracies in fairness_metrics['age_group_accuracies'].items():
        plt.plot(epochs, accuracies, marker='o', markersize=3, linestyle='-', label=f'{age}')
    
    plt.title('Age Group Accuracies During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    
    # Add phase separator
    if len(epochs) > PHASE1_EPOCHS:
        plt.axvline(x=PHASE1_EPOCHS, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'age_accuracy_trends.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Plot emotion accuracies with handling for empty arrays
    plt.figure(figsize=(12, 6))
    plotted_emotions = False
    
    for emotion, accuracies_list in fairness_metrics['emotion_accuracies'].items():
        # Skip emotions with no data
        if len(accuracies_list) == 0:
            print(f"Warning: No accuracy data for emotion '{emotion}', skipping in plot.")
            continue
            
        # Pad shorter arrays to match epochs length
        if len(accuracies_list) < len(epochs):
            pad_length = len(epochs) - len(accuracies_list)
            pad_value = 0.0 if not accuracies_list else accuracies_list[-1]
            padded_accuracies = list(accuracies_list) + [pad_value] * pad_length
            print(f"Warning: Padding emotion '{emotion}' accuracies from length {len(accuracies_list)} to {len(epochs)}")
            plt.plot(epochs, padded_accuracies, marker='o', markersize=3, linestyle='-', label=f'{emotion}')
            plotted_emotions = True
        else:
            plt.plot(epochs, accuracies_list, marker='o', markersize=3, linestyle='-', label=f'{emotion}')
            plotted_emotions = True
    
    if not plotted_emotions:
        # If no emotions were plotted, add a dummy plot
        plt.text(0.5, 0.5, "No emotion accuracy data available", 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().get_xaxis_transform())
    
    plt.title('Emotion Accuracies During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if plotted_emotions:
        plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.0)
    
    # Add phase separator
    if len(epochs) > PHASE1_EPOCHS:
        plt.axvline(x=PHASE1_EPOCHS, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'emotion_accuracy_trends.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Save fairness data to CSV
    fairness_df = pd.DataFrame({
        'epoch': epochs,
        'gender_fairness': fairness_metrics['gender_fairness_scores'],
        'age_fairness': fairness_metrics['age_fairness_scores'],
        'emotion_fairness': fairness_metrics['emotion_fairness_scores']
    })
    
    # Add demographic group accuracies
    for gender, accuracies in fairness_metrics['gender_group_accuracies'].items():
        fairness_df[f'gender_{gender.lower().replace(" ", "_")}'] = accuracies
        
    for age, accuracies in fairness_metrics['age_group_accuracies'].items():
        age_key = age.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        fairness_df[f'age_{age_key}'] = accuracies
    
    # Add emotion accuracies (only for emotions with data)
    for emotion, accuracies in fairness_metrics['emotion_accuracies'].items():
        if len(accuracies) > 0:
            emotion_key = emotion.lower()
            # Handle shorter accuracies arrays
            if len(accuracies) < len(epochs):
                pad_length = len(epochs) - len(accuracies)
                pad_value = 0.0 if not accuracies else accuracies[-1]
                padded_accuracies = list(accuracies) + [pad_value] * pad_length
                fairness_df[f'emotion_{emotion_key}'] = padded_accuracies
            else:
                fairness_df[f'emotion_{emotion_key}'] = accuracies
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'fairness_trends.csv')
    fairness_df.to_csv(csv_path, index=False)
    print(f"Fairness trends saved to {output_dir}")

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        title: Plot title
        save_path: Path to save the plot
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
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def plot_group_accuracies(group_metrics, title='Group Accuracies', save_path=None):
    """
    Plot accuracies by demographic group
    
    Args:
        group_metrics: Dictionary with group accuracy information
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Extract group accuracies
    groups = []
    accuracies = []
    
    for group, acc in group_metrics['group_accuracies'].items():
        groups.append(group)
        accuracies.append(acc * 100)  # Convert to percentage
    
    # Plot bar chart
    bars = plt.bar(groups, accuracies)
    plt.title(f"{title}\nFairness Score: {group_metrics['fairness_score']:.4f}")
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{height:.2f}%',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Group accuracies saved to {save_path}")
    
    plt.close()

def plot_emotion_by_group(group_metrics, save_path=None):
    """
    Plot emotion recognition accuracies by demographic group
    
    Args:
        group_metrics: Dictionary with emotion-by-group information
        save_path: Path to save the plot
    """
    # Extract emotion-by-group data
    emotion_by_group = group_metrics['emotion_by_group']
    
    # Convert to DataFrame for easier plotting
    data = []
    for group, emotions in emotion_by_group.items():
        for emotion, acc in emotions.items():
            data.append({
                'Group': group,
                'Emotion': emotion,
                'Accuracy': acc * 100  # Convert to percentage
            })
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot_df = df.pivot(index='Emotion', columns='Group', values='Accuracy')
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        vmin=0,
        vmax=100
    )
    plt.title(f"Emotion Recognition Accuracy by {group_metrics['attribute']} (%)")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Emotion by group heatmap saved to {save_path}")
    
    plt.close()

def plot_demographic_equalized_odds(equalized_odds_metrics, title='Equalized Odds', save_path=None):
    """
    Plot equalized odds metrics (TPR and FPR differences)
    
    Args:
        equalized_odds_metrics: Dictionary with equalized odds metrics
        title: Plot title
        save_path: Path to save the plot
    """
    # Extract metrics for each emotion
    emotions = []
    tpr_diffs = []
    fpr_diffs = []
    scores = []
    
    for emotion, metrics in equalized_odds_metrics.items():
        emotions.append(emotion)
        tpr_diffs.append(metrics['tpr_difference'] * 100)  # Convert to percentage
        fpr_diffs.append(metrics['fpr_difference'] * 100)  # Convert to percentage
        scores.append(metrics['equalized_odds_score'] * 100)  # Convert to percentage
    
    # Set up figure
    plt.figure(figsize=(14, 6))
    
    # Plot differences
    x = np.arange(len(emotions))
    width = 0.25
    
    plt.bar(x - width, tpr_diffs, width, label='TPR Difference', color='skyblue')
    plt.bar(x, fpr_diffs, width, label='FPR Difference', color='salmon')
    plt.bar(x + width, scores, width, label='Equalized Odds Score', color='lightgreen')
    
    plt.xlabel('Emotion')
    plt.ylabel('Percentage (%)')
    plt.title(title)
    plt.xticks(x, emotions, rotation=45)
    plt.legend()
    
    # Add value labels
    for i, emotion in enumerate(emotions):
        plt.text(i - width, tpr_diffs[i] + 0.5, f'{tpr_diffs[i]:.1f}%', 
                 ha='center', va='bottom', fontsize=8)
        plt.text(i, fpr_diffs[i] + 0.5, f'{fpr_diffs[i]:.1f}%', 
                 ha='center', va='bottom', fontsize=8)
        plt.text(i + width, scores[i] + 0.5, f'{scores[i]:.1f}%', 
                 ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Equalized odds plot saved to {save_path}")
    
    plt.close()

def plot_intersectional_heatmap(intersectional_metrics, title="Intersectional Fairness", save_path=None):
    """
    Plot intersectional accuracy heatmap
    
    Args:
        intersectional_metrics: Dictionary with intersectional accuracy information
        title: Plot title
        save_path: Path to save the plot
    """
    # Extract intersectional accuracies
    intersectional_accuracies = intersectional_metrics['intersectional_accuracies']
    
    if not intersectional_accuracies:
        print("No intersectional data available to plot")
        return
    
    # Convert to DataFrame
    data = []
    for group_name, metrics in intersectional_accuracies.items():
        # Split group name into gender and age
        gender, age = group_name.split(' - ')
        data.append({
            'Gender': gender,
            'Age': age,
            'Accuracy': metrics['accuracy'] * 100,  # Convert to percentage
            'Count': metrics['count']
        })
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot_df = df.pivot(index='Age', columns='Gender', values='Accuracy')
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        vmin=0,
        vmax=100
    )
    plt.title(f"{title}\nFairness Score: {intersectional_metrics['fairness_score']:.4f}")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Intersectional heatmap saved to {save_path}")
    
    plt.close()
    
    # Also create a heatmap of sample counts
    count_pivot = df.pivot(index='Age', columns='Gender', values='Count')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        count_pivot,
        annot=True,
        fmt='d',
        cmap='Greens'
    )
    plt.title(f"Sample Counts for {title}")
    plt.tight_layout()
    
    if save_path:
        count_path = save_path.replace('.png', '_counts.png')
        plt.savefig(count_path, dpi=300)
        print(f"Intersectional count heatmap saved to {count_path}")
    
    plt.close()

def visualize_evaluation_results(results, output_dir=None):
    """
    Create comprehensive visualization of evaluation results
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Directory to save visualizations
    """
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, 'evaluation')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    if 'emotion_metrics' in results and 'confusion_matrix' in results['emotion_metrics']:
        cm = results['emotion_metrics']['confusion_matrix']
        plot_confusion_matrix(
            cm,
            list(FERPLUS_EMOTIONS.values()),
            title='Emotion Recognition Confusion Matrix',
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
    
    # Plot gender group accuracies
    if 'gender_fairness' in results:
        plot_group_accuracies(
            results['gender_fairness'],
            title='Emotion Recognition Accuracy by Gender',
            save_path=os.path.join(output_dir, 'gender_accuracies.png')
        )
        
        # Plot emotion by gender
        plot_emotion_by_group(
            results['gender_fairness'],
            save_path=os.path.join(output_dir, 'emotion_by_gender.png')
        )
        
        # Plot equalized odds for gender
        if 'equalized_odds' in results['gender_fairness']:
            plot_demographic_equalized_odds(
                results['gender_fairness']['equalized_odds'],
                title='Gender Equalized Odds by Emotion',
                save_path=os.path.join(output_dir, 'gender_equalized_odds.png')
            )
    
    # Plot age group accuracies
    if 'age_fairness' in results:
        plot_group_accuracies(
            results['age_fairness'],
            title='Emotion Recognition Accuracy by Age Group',
            save_path=os.path.join(output_dir, 'age_accuracies.png')
        )
        
        # Plot emotion by age
        plot_emotion_by_group(
            results['age_fairness'],
            save_path=os.path.join(output_dir, 'emotion_by_age.png')
        )
        
        # Plot equalized odds for age
        if 'equalized_odds' in results['age_fairness']:
            plot_demographic_equalized_odds(
                results['age_fairness']['equalized_odds'],
                title='Age Equalized Odds by Emotion',
                save_path=os.path.join(output_dir, 'age_equalized_odds.png')
            )
    
    # Plot intersectional fairness
    if 'intersectional_metrics' in results:
        plot_intersectional_heatmap(
            results['intersectional_metrics'],
            title='Intersectional Fairness (Gender × Age)',
            save_path=os.path.join(output_dir, 'intersectional_fairness.png')
        )
    
    # Plot emotion class accuracies
    if 'emotion_metrics' in results and 'class_accuracies' in results['emotion_metrics']:
        class_accuracies = results['emotion_metrics']['class_accuracies']
        
        plt.figure(figsize=(12, 6))
        
        # Plot bar chart
        emotions = list(class_accuracies.keys())
        accuracies = [acc * 100 for acc in class_accuracies.values()]
        
        bars = plt.bar(emotions, accuracies)
        plt.title(f"Emotion Recognition Accuracy by Class\nFairness Score: {results['emotion_metrics']['emotion_fairness']:.4f}")
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{height:.2f}%',
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_accuracies.png'), dpi=300)
        plt.close()
    
    # Create summary chart
    plt.figure(figsize=(10, 6))
    
    metrics = []
    values = []
    
    # Add overall accuracy
    if 'emotion_metrics' in results and 'accuracy' in results['emotion_metrics']:
        metrics.append('Overall Accuracy')
        values.append(results['emotion_metrics']['accuracy'] * 100)
    
    # Add emotion fairness
    if 'emotion_metrics' in results and 'emotion_fairness' in results['emotion_metrics']:
        metrics.append('Emotion Fairness')
        values.append(results['emotion_metrics']['emotion_fairness'] * 100)
    
    # Add gender fairness
    if 'gender_fairness' in results and 'fairness_score' in results['gender_fairness']:
        metrics.append('Gender Fairness')
        values.append(results['gender_fairness']['fairness_score'] * 100)
    
    # Add age fairness
    if 'age_fairness' in results and 'fairness_score' in results['age_fairness']:
        metrics.append('Age Fairness')
        values.append(results['age_fairness']['fairness_score'] * 100)
    
    # Add intersectional fairness
    if 'intersectional_metrics' in results and 'fairness_score' in results['intersectional_metrics']:
        metrics.append('Intersectional Fairness')
        values.append(results['intersectional_metrics']['fairness_score'] * 100)
    
    # Add auxiliary task accuracies
    if 'gender_accuracy' in results:
        metrics.append('Gender Accuracy')
        values.append(results['gender_accuracy'] * 100)
    
    if 'age_accuracy' in results:
        metrics.append('Age Accuracy')
        values.append(results['age_accuracy'] * 100)
    
    # Plot bar chart
    bars = plt.bar(metrics, values)
    plt.title('Multi-Task Model Performance Summary')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{height:.2f}%',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=300)
    plt.close()
    
    print(f"Evaluation visualizations saved to {output_dir}")