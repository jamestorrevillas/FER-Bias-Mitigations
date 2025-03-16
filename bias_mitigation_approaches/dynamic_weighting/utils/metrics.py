# bias_mitigation_approaches/dynamic_weighting/utils/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from collections import defaultdict
from . import config

def softmax(x):
    """
    Compute softmax values for each set of scores in x
    
    Args:
        x: Input tensor
        
    Returns:
        Softmax activations
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def calculate_emotion_accuracies(y_true, y_pred, emotion_mapping=None):
    """
    Calculate accuracy for each emotion category
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        emotion_mapping: Optional mapping from indices to emotion names
    
    Returns:
        Dictionary of emotion accuracies
    """
    if emotion_mapping is None:
        emotion_mapping = config.FERPLUS_EMOTIONS
    
    emotion_accuracies = {}
    
    # Find unique emotion labels in the true labels
    unique_emotions = np.unique(y_true)
    
    for emotion_idx in unique_emotions:
        if emotion_idx in emotion_mapping:
            # Find samples of this emotion
            mask = y_true == emotion_idx
            if np.sum(mask) > 0:  # Avoid division by zero
                # Calculate accuracy for this emotion
                accuracy = np.mean(y_pred[mask] == emotion_idx)
                emotion_name = emotion_mapping[emotion_idx]
                emotion_accuracies[emotion_name] = accuracy
    
    return emotion_accuracies

def calculate_demographic_metrics(y_true, y_pred, demographics, demo_mapping, attribute_name):
    """
    Calculate metrics broken down by demographic attribute
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        demographics: Array of demographic attributes
        demo_mapping: Mapping from attribute codes to names
        attribute_name: Name of the attribute (e.g., "Gender", "Age")
    
    Returns:
        Dictionary containing fairness metrics
    """
    # Calculate accuracies per demographic group
    group_accuracies = {}
    group_f1_scores = {}
    group_samples = {}
    
    for demo_val in np.unique(demographics):
        if demo_val in demo_mapping:
            mask = demographics == demo_val
            if np.any(mask):
                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]
                
                # Calculate accuracy
                accuracy = np.mean(group_y_pred == group_y_true)
                
                # Calculate F1 score (more robust for imbalanced classes)
                # Use macro averaging to treat all classes equally
                try:
                    f1 = f1_score(group_y_true, group_y_pred, average='macro')
                except:
                    f1 = accuracy  # Fallback if F1 calculation fails
                
                # Store results
                demo_name = demo_mapping[demo_val]
                group_accuracies[demo_name] = accuracy
                group_f1_scores[demo_name] = f1
                group_samples[demo_name] = np.sum(mask)
    
    # Calculate fairness score as min/max ratio
    if group_f1_scores:
        min_f1 = min(group_f1_scores.values())
        max_f1 = max(group_f1_scores.values())
        fairness_score = min_f1 / max_f1 if max_f1 > 0 else 0
    else:
        fairness_score = 0
    
    # Calculate per-emotion metrics for each demographic group
    emotion_by_demo = {}
    for demo_val, demo_name in demo_mapping.items():
        demo_mask = demographics == demo_val
        emotion_by_demo[demo_name] = {}
        
        for emotion_idx in np.unique(y_true):
            # Skip if not in mapping
            if emotion_idx not in config.FERPLUS_TO_RAFDB or config.FERPLUS_TO_RAFDB[emotion_idx] is None:
                continue
                
            emotion_mask = y_true == emotion_idx
            combined_mask = demo_mask & emotion_mask
            
            if np.any(combined_mask) and np.sum(combined_mask) >= config.MIN_GROUP_SAMPLES:
                # Calculate accuracy for this demographic-emotion combination
                accuracy = np.mean(y_pred[combined_mask] == emotion_idx)
                
                # Map to emotion name
                if emotion_idx in config.RAFDB_EMOTIONS:
                    emotion_name = config.RAFDB_EMOTIONS[emotion_idx]
                else:
                    emotion_name = f"Emotion_{emotion_idx}"
                
                emotion_by_demo[demo_name][emotion_name] = accuracy
    
    return {
        'attribute': attribute_name,
        'fairness_score': fairness_score,
        'accuracies': group_accuracies,
        'f1_scores': group_f1_scores,
        'sample_counts': group_samples,
        'emotion_by_demo': emotion_by_demo
    }

def calculate_emotion_fairness(emotion_accuracies):
    """
    Calculate emotion fairness score based on emotion accuracies
    
    Args:
        emotion_accuracies: Dictionary of emotion accuracies
    
    Returns:
        Fairness score (min/max accuracy ratio)
    """
    if not emotion_accuracies:
        return 0
    
    min_acc = min(emotion_accuracies.values())
    max_acc = max(emotion_accuracies.values())
    
    if max_acc > 0:
        return min_acc / max_acc
    else:
        return 0

def map_ferplus_to_rafdb_predictions(ferplus_predictions):
    """
    Map FER+ emotion indices to RAF-DB emotion indices
    
    Args:
        ferplus_predictions: Array of FER+ emotion predictions
        
    Returns:
        Array of RAF-DB emotion predictions
    """
    rafdb_predictions = []
    
    for pred in ferplus_predictions:
        if pred in config.FERPLUS_TO_RAFDB and config.FERPLUS_TO_RAFDB[pred] is not None:
            rafdb_predictions.append(config.FERPLUS_TO_RAFDB[pred])
        else:
            # Map contempt to neutral as fallback
            rafdb_predictions.append(7)  # Neutral in RAF-DB
    
    return np.array(rafdb_predictions)

def identify_underperforming_groups(metrics_result):
    """
    Identify demographic groups that are performing poorly
    
    Args:
        metrics_result: Output from calculate_demographic_metrics
        
    Returns:
        Dictionary mapping demographic groups to their performance gap
    """
    performance_gaps = {}
    
    # Find best performing group
    best_score = max(metrics_result['f1_scores'].values())
    
    # Calculate gap for each group
    for group, score in metrics_result['f1_scores'].items():
        gap = best_score - score
        performance_gaps[group] = gap
    
    return performance_gaps

def identify_underperforming_intersections(metrics_result, threshold=0.2):
    """
    Identify demographic-emotion intersections with poor performance
    
    Args:
        metrics_result: Output from calculate_demographic_metrics
        threshold: Performance gap threshold to consider underperforming
        
    Returns:
        List of tuples (demographic, emotion, performance_gap)
    """
    underperforming = []
    
    # Extract emotion-by-demo data
    emotion_by_demo = metrics_result['emotion_by_demo']
    
    # Find best performance for each emotion across demographics
    best_by_emotion = {}
    
    for demo, emotions in emotion_by_demo.items():
        for emotion, accuracy in emotions.items():
            if emotion not in best_by_emotion or accuracy > best_by_emotion[emotion]:
                best_by_emotion[emotion] = accuracy
    
    # Find gaps
    for demo, emotions in emotion_by_demo.items():
        for emotion, accuracy in emotions.items():
            if emotion in best_by_emotion:
                gap = best_by_emotion[emotion] - accuracy
                if gap >= threshold:
                    underperforming.append((demo, emotion, gap))
    
    # Sort by gap, largest first
    underperforming.sort(key=lambda x: x[2], reverse=True)
    
    return underperforming

def evaluate_model_fairness(model, rafdb_images, rafdb_labels, demographic_info):
    """
    Evaluate model fairness on RAF-DB dataset
    
    Args:
        model: Trained model
        rafdb_images: RAF-DB image data
        rafdb_labels: RAF-DB labels
        demographic_info: Demographic information
        
    Returns:
        Dictionary of fairness metrics
    """
    # Extract demographic attributes
    gender_attributes = np.array([info['gender'] for info in demographic_info])
    age_attributes = np.array([info['age'] for info in demographic_info])
    
    # Generate predictions
    y_pred_logits = model.predict(rafdb_images)
    y_pred_probs = softmax(y_pred_logits)
    
    # Get predicted FER+ classes
    ferplus_pred = np.argmax(y_pred_probs, axis=1)
    
    # Map to RAF-DB emotion indices
    rafdb_pred = map_ferplus_to_rafdb_predictions(ferplus_pred)
    
    # Calculate demographic metrics
    gender_metrics = calculate_demographic_metrics(
        rafdb_labels, rafdb_pred, gender_attributes, 
        config.GENDER_LABELS, "Gender"
    )
    
    age_metrics = calculate_demographic_metrics(
        rafdb_labels, rafdb_pred, age_attributes, 
        config.AGE_GROUPS, "Age"
    )
    
    # Calculate overall accuracy
    overall_accuracy = np.mean(rafdb_pred == rafdb_labels)
    
    # Calculate emotion accuracies
    emotion_accuracies = {}
    for raf_label, emotion_name in config.RAFDB_EMOTIONS.items():
        mask = rafdb_labels == raf_label
        if np.any(mask):
            acc = np.mean(rafdb_pred[mask] == raf_label)
            emotion_accuracies[emotion_name] = acc
    
    # Calculate emotion fairness
    emotion_fairness = calculate_emotion_fairness(emotion_accuracies)
    
    return {
        'overall_accuracy': overall_accuracy,
        'gender_metrics': gender_metrics,
        'age_metrics': age_metrics,
        'emotion_accuracies': emotion_accuracies,
        'emotion_fairness': emotion_fairness
    }

def compute_weight_distribution(sample_weights):
    """
    Compute statistics about the sample weight distribution
    
    Args:
        sample_weights: Array of sample weights
        
    Returns:
        Dictionary with weight distribution statistics
    """
    return {
        'mean': np.mean(sample_weights),
        'median': np.median(sample_weights),
        'min': np.min(sample_weights),
        'max': np.max(sample_weights),
        'std': np.std(sample_weights),
        'histogram': np.histogram(sample_weights, bins=10)[0].tolist()
    }