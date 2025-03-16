# utils/comparative_analysis/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_recall_fscore_support

# Updated mapping dictionaries to match your data
RAFDB_EMOTIONS = {
    1: 'Surprise',
    2: 'Fear',
    3: 'Disgust',
    4: 'Happy',
    5: 'Sad',
    6: 'Angry',
    7: 'Neutral'
}

FERPLUS_EMOTIONS = {
    0: 'Neutral',
    1: 'Happiness',
    2: 'Surprise',
    3: 'Sadness',
    4: 'Anger', 
    5: 'Disgust',
    6: 'Fear',
    7: 'Contempt'
}

RAFDB_TO_FERPLUS = {
    1: 2,  # Surprise -> Surprise
    2: 6,  # Fear -> Fear
    3: 5,  # Disgust -> Disgust
    4: 1,  # Happy -> Happiness
    5: 3,  # Sad -> Sadness
    6: 4,  # Angry -> Anger
    7: 0   # Neutral -> Neutral
}

# Updated to match your dataset coding
GENDER_LABELS = {
    0: 'Male',
    1: 'Female'
}

AGE_GROUPS = {
    1: 'Child (0-12)',
    2: 'Teen (13-19)',
    3: 'Young Adult (20-40)',
    4: 'Adult (41-60)',
    5: 'Senior (60+)'
}

def calculate_overall_metrics(y_true, y_pred):
    """Calculate overall performance metrics including F1-score"""
    accuracy = accuracy_score(y_true, y_pred)
    # Calculate macro F1-score (average of F1 for each class, giving equal weight)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Calculate precision and recall metrics as well
    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'report': report
    }

def calculate_fairness_metrics(predictions, labels, demographic_group):
    """Calculate fairness metrics across demographic groups using F1-score"""
    unique_groups = np.unique(demographic_group)
    accuracies = {}
    f1_scores = {}

    for group in unique_groups:
        mask = demographic_group == group
        group_preds = predictions[mask]
        group_labels = labels[mask]
        if len(group_labels) > 0 and len(np.unique(group_labels)) > 1:
            # Calculate F1-score for this demographic group
            f1 = f1_score(group_labels, group_preds, average='macro', zero_division=0)
            f1_scores[group] = f1
            # Keep accuracy for reference
            accuracies[group] = np.mean(group_preds == group_labels)
        else:
            f1_scores[group] = 0
            accuracies[group] = 0

    # Calculate fairness score based on F1-scores (minimum ratio of F1-scores)
    max_f1 = max(f1_scores.values()) if f1_scores else 0
    if max_f1 > 0:
        fairness_score = min(f1/max_f1 for f1 in f1_scores.values() if f1 > 0)
    else:
        fairness_score = 0

    return fairness_score, accuracies, f1_scores

def calculate_emotion_accuracies(y_true, y_pred):
    """Calculate F1-score for each emotion category"""
    emotion_metrics = {}
    
    for raf_label, emotion_name in RAFDB_EMOTIONS.items():
        mask = y_true == raf_label
        if np.any(mask):
            # Get the corresponding FER+ label
            fer_label = RAFDB_TO_FERPLUS[raf_label]
            
            # Get true and predicted labels for this emotion
            y_true_bin = np.zeros_like(y_true[mask], dtype=int)  # Initialize with all 0s
            y_pred_bin = (y_pred[mask] == fer_label).astype(int)  # 1 for correct, 0 for incorrect predictions
            
            # For binary F1-score calculation, create true binary classes
            # This is a binary classification where 1 = correct prediction
            if np.sum(y_pred_bin) > 0:  # If there are any positive predictions
                # Calculate precision: TP / (TP + FP)
                # Here all predictions are for this emotion, so precision is TP / total predictions
                precision = np.sum(y_pred_bin) / len(y_pred_bin)
                
                # For recall, we know all samples are this emotion, so recall is TP / total samples
                recall = np.sum(y_pred_bin) / len(y_true_bin)
                
                # Calculate F1-score
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0
            else:
                f1 = 0
            
            # Also calculate accuracy for reference
            acc = np.mean(y_pred[mask] == fer_label)
            
            emotion_metrics[emotion_name] = {
                'f1_score': f1,
                'accuracy': acc
            }
    
    return emotion_metrics

def calculate_demographic_metrics(y_true, y_pred, demographics, demo_mapping, attribute_name):
    """Calculate metrics broken down by demographic attribute using F1-score"""
    # Calculate F1-scores per demographic group
    group_metrics = {}
    
    for demo_val in demo_mapping.keys():
        mask = demographics == demo_val
        if np.any(mask):
            # Convert RAF-DB labels to FER+ labels for this specific group
            y_true_masked = y_true[mask]
            y_true_fer_group = np.array([RAFDB_TO_FERPLUS[label] for label in y_true_masked 
                                        if label in RAFDB_TO_FERPLUS])
            
            if len(y_true_fer_group) > 0 and len(np.unique(y_true_fer_group)) > 1:
                # Truncate predictions to match true values if needed
                y_pred_group = y_pred[mask][:len(y_true_fer_group)]
                
                # Calculate F1-score for this demographic group
                f1 = f1_score(y_true_fer_group, y_pred_group, average='macro', zero_division=0)
                
                # Calculate accuracy for reference
                acc = np.mean(y_pred_group == y_true_fer_group)
                
                group_metrics[demo_val] = {
                    'f1_score': f1,
                    'accuracy': acc
                }
            else:
                group_metrics[demo_val] = {
                    'f1_score': 0,
                    'accuracy': 0
                }
    
    # Calculate fairness score based on F1-scores
    if group_metrics:
        f1_scores = {k: v['f1_score'] for k, v in group_metrics.items()}
        min_f1 = min(f1_scores.values())
        max_f1 = max(f1_scores.values())
        fairness_score = min_f1 / max_f1 if max_f1 > 0 else 0
    else:
        fairness_score = 0
    
    # Calculate per-emotion metrics for each demographic group
    emotion_by_demo = {}
    for demo_val, demo_name in demo_mapping.items():
        demo_mask = demographics == demo_val
        emotion_by_demo[demo_name] = {}
        
        for raf_label, emotion_name in RAFDB_EMOTIONS.items():
            emotion_mask = y_true == raf_label
            combined_mask = demo_mask & emotion_mask
            
            if np.any(combined_mask):
                fer_label = RAFDB_TO_FERPLUS[raf_label]
                
                # Binary prediction for this demographic + emotion combination
                y_pred_bin = (y_pred[combined_mask] == fer_label).astype(int)
                
                # Calculate accuracy
                acc = np.mean(y_pred[combined_mask] == fer_label)
                
                # For F1-score, we treat this as a binary classification problem
                # All samples in combined_mask have the same true emotion
                if len(y_pred_bin) > 0:
                    # Precision: how many of our predictions were correct
                    precision = np.sum(y_pred_bin) / len(y_pred_bin) if np.sum(y_pred_bin) > 0 else 0
                    
                    # Recall: how many of the true cases we identified
                    # In this case, recall = accuracy because all samples should be correctly predicted
                    recall = acc
                    
                    # F1-score
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1 = 0
                else:
                    f1 = 0
                
                emotion_by_demo[demo_name][emotion_name] = {
                    'f1_score': f1, 
                    'accuracy': acc
                }
    
    return {
        'attribute': attribute_name,
        'fairness_score': fairness_score,
        'metrics': {demo_mapping[k]: v for k, v in group_metrics.items()},
        'emotion_by_demo': emotion_by_demo
    }