# bias_mitigation_approaches/multitask_learning/utils/metrics.py

import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import sys

# Append parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bias_mitigation_approaches.multitask_learning.utils.config import *

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

def calculate_fairness_score(group_accuracies):
    """
    Calculate fairness score as the ratio of minimum to maximum accuracy across groups
    
    Args:
        group_accuracies: Dictionary mapping group names to accuracies
        
    Returns:
        Fairness score (0-1, higher is better)
    """
    if not group_accuracies:
        return 0
    
    min_acc = min(group_accuracies.values())
    max_acc = max(group_accuracies.values())
    
    if max_acc > 0:
        return min_acc / max_acc
    else:
        return 0

def calculate_demographic_parity(y_pred, protected_attribute, positive_class=None):
    """
    Calculate demographic parity difference - measures equal prediction rates
    across different demographic groups
    
    Args:
        y_pred: Predicted labels
        protected_attribute: Protected attribute values for each sample
        positive_class: Optional value representing positive prediction
        
    Returns:
        Demographic parity difference and ratio
    """
    if positive_class is None:
        # If no positive class specified, assume all predictions are probabilities
        # and use threshold of 0.5
        if np.min(y_pred) >= 0 and np.max(y_pred) <= 1:
            y_pred_binary = (y_pred >= 0.5).astype(int)
        else:
            # Otherwise, assume all non-zero predictions are positive
            y_pred_binary = (y_pred != 0).astype(int)
    else:
        y_pred_binary = (y_pred == positive_class).astype(int)
    
    # Calculate prediction rates by group
    group_0_indices = np.where(protected_attribute == 0)[0]
    group_1_indices = np.where(protected_attribute == 1)[0]
    
    if len(group_0_indices) == 0 or len(group_1_indices) == 0:
        return 0, 1  # No difference if only one group is present
    
    prediction_rate_0 = np.mean(y_pred_binary[group_0_indices])
    prediction_rate_1 = np.mean(y_pred_binary[group_1_indices])
    
    # Calculate difference and ratio
    difference = abs(prediction_rate_0 - prediction_rate_1)
    
    # Calculate ratio for comparison
    if prediction_rate_0 > 0 and prediction_rate_1 > 0:
        ratio = min(prediction_rate_0, prediction_rate_1) / max(prediction_rate_0, prediction_rate_1)
    else:
        ratio = 0
    
    return difference, ratio

def calculate_equalized_odds(y_true, y_pred, protected_attribute, positive_class=None):
    """
    Calculate equalized odds metric - measures equal error rates across groups
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        protected_attribute: Protected attribute values for each sample
        positive_class: Optional value representing positive prediction
        
    Returns:
        Dictionary with equalized odds metrics
    """
    # Convert to binary classification problem
    if positive_class is None:
        # If no positive class specified, assume binary classification
        y_true_binary = y_true
        y_pred_binary = y_pred
    else:
        y_true_binary = (y_true == positive_class).astype(int)
        y_pred_binary = (y_pred == positive_class).astype(int)
    
    # Split data by demographic group
    group_0_indices = np.where(protected_attribute == 0)[0]
    group_1_indices = np.where(protected_attribute == 1)[0]
    
    if len(group_0_indices) == 0 or len(group_1_indices) == 0:
        return {
            'tpr_difference': 0,
            'fpr_difference': 0,
            'equalized_odds_score': 1.0
        }
    
    # Calculate true positive rates (TPR) for each group
    true_positives_0 = np.sum((y_true_binary[group_0_indices] == 1) & 
                             (y_pred_binary[group_0_indices] == 1))
    positives_0 = np.sum(y_true_binary[group_0_indices] == 1)
    tpr_0 = true_positives_0 / positives_0 if positives_0 > 0 else 0
    
    true_positives_1 = np.sum((y_true_binary[group_1_indices] == 1) & 
                             (y_pred_binary[group_1_indices] == 1))
    positives_1 = np.sum(y_true_binary[group_1_indices] == 1)
    tpr_1 = true_positives_1 / positives_1 if positives_1 > 0 else 0
    
    # Calculate false positive rates (FPR) for each group
    false_positives_0 = np.sum((y_true_binary[group_0_indices] == 0) & 
                              (y_pred_binary[group_0_indices] == 1))
    negatives_0 = np.sum(y_true_binary[group_0_indices] == 0)
    fpr_0 = false_positives_0 / negatives_0 if negatives_0 > 0 else 0
    
    false_positives_1 = np.sum((y_true_binary[group_1_indices] == 0) & 
                              (y_pred_binary[group_1_indices] == 1))
    negatives_1 = np.sum(y_true_binary[group_1_indices] == 0)
    fpr_1 = false_positives_1 / negatives_1 if negatives_1 > 0 else 0
    
    # Calculate differences
    tpr_difference = abs(tpr_0 - tpr_1)
    fpr_difference = abs(fpr_0 - fpr_1)
    
    # Calculate equalized odds score (higher is better)
    equalized_odds_score = 1.0 - (tpr_difference + fpr_difference) / 2.0
    
    return {
        'tpr_difference': tpr_difference,
        'fpr_difference': fpr_difference,
        'tpr_0': tpr_0,
        'tpr_1': tpr_1,
        'fpr_0': fpr_0,
        'fpr_1': fpr_1,
        'equalized_odds_score': equalized_odds_score
    }

def calculate_intersectional_metrics(y_true, y_pred, gender_attr, age_attr, 
                                   gender_mapping=None, age_mapping=None):
    """
    Calculate metrics for intersectional demographic groups
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        gender_attr: Gender attribute values
        age_attr: Age attribute values
        gender_mapping: Mapping of gender codes to names
        age_mapping: Mapping of age codes to names
        
    Returns:
        Dictionary with intersectional metrics
    """
    if gender_mapping is None:
        gender_mapping = GENDER_LABELS
    
    if age_mapping is None:
        age_mapping = AGE_GROUPS
    
    # Initialize results
    intersectional_accuracies = {}
    
    # Calculate accuracy for each intersectional group
    for gender_code, gender_name in gender_mapping.items():
        for age_code, age_name in age_mapping.items():
            # Find samples in this intersectional group
            if isinstance(age_attr, np.ndarray) and isinstance(gender_attr, np.ndarray):
                # Handle case where age groups are 1-indexed but array indices are 0-indexed
                adj_age_code = age_code - 1 if age_code >= 1 else age_code
                group_indices = np.where((gender_attr == gender_code) & 
                                        (age_attr == adj_age_code))[0]
            else:
                # Handle case where attributes are lists of dictionaries
                group_indices = [i for i in range(len(gender_attr)) 
                               if gender_attr[i] == gender_code and age_attr[i] == age_code]
            
            if len(group_indices) > 0:
                # Calculate accuracy for this group
                group_acc = np.mean(np.array(y_pred)[group_indices] == np.array(y_true)[group_indices])
                group_name = f"{gender_name} - {age_name}"
                intersectional_accuracies[group_name] = {
                    'accuracy': group_acc,
                    'count': len(group_indices),
                    'gender_code': gender_code,
                    'age_code': age_code
                }
    
    # Calculate fairness score across all intersectional groups
    if intersectional_accuracies:
        acc_values = [group['accuracy'] for group in intersectional_accuracies.values()]
        min_acc = min(acc_values)
        max_acc = max(acc_values)
        fairness_score = min_acc / max_acc if max_acc > 0 else 0
    else:
        fairness_score = 0
    
    return {
        'intersectional_accuracies': intersectional_accuracies,
        'fairness_score': fairness_score
    }

def evaluate_emotion_recognition(y_true, y_pred, emotion_mapping=None):
    """
    Evaluate emotion recognition performance
    
    Args:
        y_true: True emotion labels
        y_pred: Predicted emotion labels
        emotion_mapping: Optional mapping from indices to emotion names
        
    Returns:
        Dictionary with evaluation metrics
    """
    if emotion_mapping is None:
        emotion_mapping = FERPLUS_EMOTIONS
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class accuracy
    class_accuracies = {}
    for emotion_idx in np.unique(y_true):
        if emotion_idx in emotion_mapping:
            # Get indices of samples with this emotion
            indices = np.where(y_true == emotion_idx)[0]
            if len(indices) > 0:
                # Calculate accuracy for this emotion
                class_acc = accuracy_score(y_true[indices], y_pred[indices])
                emotion_name = emotion_mapping[emotion_idx]
                class_accuracies[emotion_name] = class_acc
    
    # Calculate emotion fairness
    emotion_fairness = calculate_fairness_score(class_accuracies)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'emotion_fairness': emotion_fairness,
        'confusion_matrix': cm
    }

def evaluate_demographic_fairness(y_true, y_pred, demographic_values, group_mapping=None, attribute_name=None):
    """
    Evaluate fairness across demographic groups with advanced metrics
    
    Args:
        y_true: True emotion labels
        y_pred: Predicted emotion labels
        demographic_values: Array of demographic attributes (gender or age)
        group_mapping: Mapping from demographic codes to names
        attribute_name: Name of the demographic attribute
        
    Returns:
        Dictionary with demographic fairness metrics
    """
    if group_mapping is None:
        if np.max(demographic_values) <= 1:
            group_mapping = GENDER_LABELS
            attribute_name = "Gender"
        else:
            group_mapping = AGE_GROUPS
            attribute_name = "Age"
    
    # Per-group accuracies
    group_accuracies = {}
    emotion_by_group = {}
    
    for group_idx in np.unique(demographic_values):
        if group_idx in group_mapping:
            # Get indices of samples in this demographic group
            indices = np.where(demographic_values == group_idx)[0]
            if len(indices) > 0:
                # Calculate accuracy for this group
                group_acc = accuracy_score(y_true[indices], y_pred[indices])
                group_name = group_mapping[group_idx]
                group_accuracies[group_name] = group_acc
                
                # Calculate per-emotion accuracy within this group
                emotion_by_group[group_name] = {}
                for emotion_idx in np.unique(y_true):
                    if emotion_idx in FERPLUS_EMOTIONS:
                        # Get indices of samples with this emotion in this group
                        emotion_indices = np.where((y_true == emotion_idx) & (demographic_values == group_idx))[0]
                        if len(emotion_indices) > 0:
                            # Calculate accuracy for this emotion in this group
                            emotion_acc = accuracy_score(y_true[emotion_indices], y_pred[emotion_indices])
                            emotion_name = FERPLUS_EMOTIONS[emotion_idx]
                            emotion_by_group[group_name][emotion_name] = emotion_acc
    
    # Calculate overall fairness score
    fairness_score = calculate_fairness_score(group_accuracies)
    
    # Calculate demographic parity for each emotion
    demographic_parity_metrics = {}
    for emotion_idx in np.unique(y_true):
        if emotion_idx in FERPLUS_EMOTIONS:
            emotion_name = FERPLUS_EMOTIONS[emotion_idx]
            y_pred_binary = (y_pred == emotion_idx).astype(int)
            
            # Calculate demographic parity for binary prediction of this emotion
            dp_diff, dp_ratio = calculate_demographic_parity(y_pred_binary, demographic_values)
            demographic_parity_metrics[emotion_name] = {
                'difference': dp_diff,
                'ratio': dp_ratio
            }
    
    # Calculate equalized odds for each emotion
    equalized_odds_metrics = {}
    for emotion_idx in np.unique(y_true):
        if emotion_idx in FERPLUS_EMOTIONS:
            emotion_name = FERPLUS_EMOTIONS[emotion_idx]
            
            # Convert to binary problem for this emotion
            y_true_binary = (y_true == emotion_idx).astype(int)
            y_pred_binary = (y_pred == emotion_idx).astype(int)
            
            # Calculate equalized odds
            eq_odds = calculate_equalized_odds(y_true_binary, y_pred_binary, demographic_values)
            equalized_odds_metrics[emotion_name] = eq_odds
    
    return {
        'attribute': attribute_name,
        'group_accuracies': group_accuracies,
        'fairness_score': fairness_score,
        'emotion_by_group': emotion_by_group,
        'demographic_parity': demographic_parity_metrics,
        'equalized_odds': equalized_odds_metrics
    }

def evaluate_multitask_model(model, test_data, test_labels):
    """
    Evaluate multi-task model performance with comprehensive fairness metrics
    
    Args:
        model: Trained multi-task model
        test_data: Test images
        test_labels: List of [emotion_labels, gender_labels, age_labels]
        
    Returns:
        Dictionary with evaluation results
    """
    # Get predictions
    predictions = model.predict(test_data)
    
    # Extract true labels
    y_true_emotion = np.argmax(test_labels[0], axis=1)
    y_true_gender = np.argmax(test_labels[1], axis=1)
    y_true_age = np.argmax(test_labels[2], axis=1)
    
    # Extract predicted labels
    y_pred_emotion = np.argmax(predictions[0], axis=1)
    y_pred_gender = np.argmax(predictions[1], axis=1)
    y_pred_age = np.argmax(predictions[2], axis=1)
    
    # Evaluate emotion recognition
    emotion_metrics = evaluate_emotion_recognition(y_true_emotion, y_pred_emotion)
    
    # Evaluate gender fairness
    gender_fairness = evaluate_demographic_fairness(
        y_true_emotion, y_pred_emotion, y_true_gender, GENDER_LABELS, "Gender"
    )
    
    # Evaluate age fairness
    age_fairness = evaluate_demographic_fairness(
        y_true_emotion, y_pred_emotion, y_true_age, AGE_GROUPS, "Age"
    )
    
    # Evaluate gender prediction accuracy
    gender_accuracy = accuracy_score(y_true_gender, y_pred_gender)
    
    # Evaluate age prediction accuracy
    age_accuracy = accuracy_score(y_true_age, y_pred_age)
    
    # Calculate intersectional fairness
    intersectional_metrics = calculate_intersectional_metrics(
        y_true_emotion, y_pred_emotion, y_true_gender, y_true_age,
        GENDER_LABELS, AGE_GROUPS
    )
    
    # Calculate AUC for auxiliary tasks if possible
    try:
        # One-hot encode true labels
        y_true_gender_onehot = np.zeros((len(y_true_gender), NUM_GENDER_CLASSES))
        y_true_gender_onehot[np.arange(len(y_true_gender)), y_true_gender] = 1
        
        y_true_age_onehot = np.zeros((len(y_true_age), NUM_AGE_CLASSES))
        y_true_age_onehot[np.arange(len(y_true_age)), y_true_age] = 1
        
        # Calculate AUC
        gender_auc = roc_auc_score(y_true_gender_onehot, predictions[1], multi_class='ovr')
        age_auc = roc_auc_score(y_true_age_onehot, predictions[2], multi_class='ovr')
    except:
        gender_auc = 0
        age_auc = 0
    
    return {
        'emotion_metrics': emotion_metrics,
        'gender_fairness': gender_fairness,
        'age_fairness': age_fairness,
        'gender_accuracy': gender_accuracy,
        'age_accuracy': age_accuracy,
        'gender_auc': gender_auc,
        'age_auc': age_auc,
        'intersectional_metrics': intersectional_metrics
    }