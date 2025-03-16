# utils/comparative_analysis/plots.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Import constants from the same directory
from metrics import (
    RAFDB_EMOTIONS, 
    FERPLUS_EMOTIONS, 
    RAFDB_TO_FERPLUS,
    AGE_GROUPS,
    GENDER_LABELS
)

def plot_emotion_accuracies(emotion_accuracies, title="Emotion Recognition Accuracy by Category", save_dir=None, show_plots=True):
    """Plot emotion-wise accuracies"""
    if save_dir is None and not show_plots:
        return  # Skip if not saving or showing
        
    plt.figure(figsize=(12, 6))
    emotions = list(emotion_accuracies.keys())
    accuracies = list(emotion_accuracies.values())

    bars = plt.bar(emotions, accuracies)
    plt.title(title)
    plt.xlabel('Emotion')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')

    plt.tight_layout()
    
    # Save the plot if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'emotion_accuracies.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_demographic_accuracies(metric_results, title=None, save_dir=None, show_plots=True):
    """Plot demographic accuracies with fairness score"""
    if save_dir is None and not show_plots:
        return  # Skip if not saving or showing
        
    fairness_score = metric_results['fairness_score']
    attribute = metric_results['attribute']
    
    # Extract metrics (use F1-scores instead of accuracies)
    metrics_dict = {}
    for group, group_metrics in metric_results.get('metrics', {}).items():
        if isinstance(group_metrics, dict) and 'f1_score' in group_metrics:
            metrics_dict[group] = group_metrics['f1_score']
    
    # If no metrics found in new format, try legacy format for backward compatibility
    if not metrics_dict and 'accuracies' in metric_results:
        metrics_dict = metric_results['accuracies']
    
    plt.figure(figsize=(12, 6))
    x = list(metrics_dict.keys())
    # Convert to percentages
    y = [acc * 100 for acc in metrics_dict.values()]

    bars = plt.bar(x, y)
    if title:
        plt.title(f'{title}\nFairness Score: {fairness_score * 100:.2f}%')
    else:
        plt.title(f'Recognition F1-Score by {attribute}\nFairness Score: {fairness_score * 100:.2f}%')
    
    plt.xlabel(attribute)
    plt.ylabel('F1-Score (%)')
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')

    plt.tight_layout()
    
    # Save the plot if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'demographic_f1scores_{attribute.lower()}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_emotion_by_demographic(metric_results, title=None, save_dir=None, show_plots=True):
    """Plot emotion accuracy breakdown by demographic group"""
    if save_dir is None and not show_plots:
        return  # Skip if not saving or showing
        
    attribute = metric_results['attribute']
    emotion_by_demo = metric_results['emotion_by_demo']
    
    # Extract F1-scores for each emotion/demographic combination
    emotion_by_demo_f1 = {}
    for demo_name, emotions in emotion_by_demo.items():
        emotion_by_demo_f1[demo_name] = {}
        for emotion_name, metrics in emotions.items():
            if isinstance(metrics, dict) and 'f1_score' in metrics:
                emotion_by_demo_f1[demo_name][emotion_name] = metrics['f1_score']
            else:
                # Backward compatibility
                emotion_by_demo_f1[demo_name][emotion_name] = metrics
    
    plt.figure(figsize=(15, 8))
    # Convert to percentages
    df = pd.DataFrame({k: {ek: ev*100 for ek, ev in v.items()} 
                      for k, v in emotion_by_demo_f1.items()})

    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', vmin=0, vmax=100)
    if title:
        plt.title(title)
    else:
        plt.title(f'Emotion Recognition F1-Score by {attribute}')
    
    plt.xlabel(attribute)
    plt.ylabel('Emotion')
    plt.tight_layout()
    
    # Save the plot if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'emotion_by_{attribute.lower()}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_dir=None, show_plots=True):
    """Plot confusion matrix"""
    if save_dir is None and not show_plots:
        return  # Skip if not saving or showing
        
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=list(FERPLUS_EMOTIONS.values()),
                yticklabels=list(FERPLUS_EMOTIONS.values()))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save the plot if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_data_distribution(test_labels, title="Dataset Distribution", save_dir=None, show_plots=True):
    """Plot the distribution of the dataset"""
    if save_dir is None and not show_plots:
        return  # Skip if not saving or showing
        
    plt.figure(figsize=(15, 5))

    # 1. Emotion Distribution
    plt.subplot(131)
    emotion_dist = test_labels['label'].value_counts().sort_index()
    valid_emotions = [i for i in emotion_dist.index if i in RAFDB_EMOTIONS]
    sns.barplot(x=[RAFDB_EMOTIONS[i] for i in valid_emotions],
               y=emotion_dist[valid_emotions].values)
    plt.title('Emotion Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')

    # 2. Gender Distribution
    plt.subplot(132)
    gender_dist = test_labels['Gender'].value_counts().sort_index()
    # Make sure we only use valid gender labels
    valid_genders = [i for i in gender_dist.index if i in GENDER_LABELS]
    sns.barplot(x=[GENDER_LABELS[i] for i in valid_genders], 
                y=[gender_dist[i] for i in valid_genders])
    plt.title('Gender Distribution')
    plt.ylabel('Count')

    # 3. Age Distribution
    plt.subplot(133)
    age_dist = test_labels['Age_Group'].value_counts().sort_index()
    # Make sure we only use valid age groups
    valid_ages = [i for i in age_dist.index if i in AGE_GROUPS]
    sns.barplot(x=[AGE_GROUPS[i] for i in valid_ages], 
                y=[age_dist[i] for i in valid_ages])
    plt.title('Age Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.85)
    
    # Save the plot if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, 'dataset_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_demographic_group_accuracies(results, attribute, save_dir=None, show_plots=True):
    """Plot detailed F1-scores for each demographic group across models"""
    if save_dir is None and not show_plots:
        return  # Skip if not saving or showing
        
    plt.figure(figsize=(14, 8))
    
    # Extract demographic data for all models
    demo_data = {}
    model_names = list(results.keys())
    
    for model_name in model_names:
        # Find the demographic metrics for this attribute
        for demo_metric in results[model_name]["demographic"]:
            if demo_metric["attribute"] == attribute:
                # Use the new metrics structure with F1-scores
                if "metrics" in demo_metric:
                    # Extract F1-scores from the new structure
                    demo_data[model_name] = {k: v['f1_score']*100 for k, v in demo_metric["metrics"].items()}
                elif "accuracies" in demo_metric:
                    # Backward compatibility with the old structure
                    demo_data[model_name] = {k: v*100 for k, v in demo_metric["accuracies"].items()}
                break
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(demo_data)
    
    # Create grouped bar chart
    ax = df.plot(kind='bar', figsize=(14, 8))
    plt.title(f'{attribute} Group F1-Scores Across Models')
    plt.xlabel(f'{attribute} Groups')
    plt.ylabel('F1-Score (%)')
    plt.ylim(0, 100.0)
    plt.xticks(rotation=45)
    plt.legend(title='Models')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%')
    
    plt.tight_layout()
    
    # Save the plot if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f'comparison_{attribute.lower()}_groups.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def plot_comparison_results(results, metric_name, save_dir=None, show_plots=True):
    """Plot comparison of different models for a specific metric"""
    if save_dir is None and not show_plots:
        return  # Skip if not saving or showing
        
    plt.figure(figsize=(12, 6))
    
    model_names = list(results.keys())
    values = []
    
    for model in model_names:
        if metric_name == "Overall Accuracy":
            # Convert to percentage
            values.append(results[model]["overall"]["accuracy"] * 100)
        elif metric_name == "Overall F1-Score":
            # Use F1-score instead of accuracy
            values.append(results[model]["overall"]["f1_score"] * 100)
        elif metric_name.startswith("Fairness-"):
            attribute = metric_name.split("-")[1]
            for demo_metric in results[model]["demographic"]:
                if demo_metric["attribute"].lower() == attribute.lower():
                    # Convert to percentage
                    values.append(demo_metric["fairness_score"] * 100)
                    break
    
    bars = plt.bar(model_names, values)
    plt.title(f'Comparison of {metric_name} Across Models')
    plt.ylabel(f'{metric_name} (%)')
    plt.ylim(0, 100.0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_metric_name = metric_name.lower().replace(' ', '_').replace('-', '_')
        filename = os.path.join(save_dir, f'comparison_{safe_metric_name}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()