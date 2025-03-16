# bias_mitigation_approaches/data_augmentation/utils/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
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
    
    for demo_val in np.unique(demographics):
        if demo_val in demo_mapping:
            mask = demographics == demo_val
            if np.any(mask):
                group_accuracies[demo_val] = np.mean(y_pred[mask] == y_true[mask])
    
    # Calculate fairness score as min/max ratio
    named_accuracies = {demo_mapping[k]: v for k, v in group_accuracies.items() 
                       if k in demo_mapping}
    
    if named_accuracies:
        min_acc = min(named_accuracies.values())
        max_acc = max(named_accuracies.values())
        fairness_score = min_acc / max_acc if max_acc > 0 else 0
    else:
        fairness_score = 0
    
    return {
        'attribute': attribute_name,
        'fairness_score': fairness_score,
        'accuracies': named_accuracies
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

def calculate_confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Calculate confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Optional number of classes
    
    Returns:
        Confusion matrix
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    
    return confusion_matrix(y_true, y_pred, labels=range(num_classes))

def print_evaluation_metrics(model, X_test, y_test, label_map=None):
    """
    Print comprehensive evaluation metrics
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels (integer format)
        label_map: Optional mapping from indices to label names
    
    Returns:
        Dictionary of evaluation results
    """
    if label_map is None:
        label_map = config.FERPLUS_EMOTIONS
    
    # Convert test labels to one-hot for evaluation with model
    y_test_onehot = np.zeros((y_test.size, config.NUM_CLASSES))
    y_test_onehot[np.arange(y_test.size), y_test] = 1
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions - handle logits output correctly
    y_pred_logits = model.predict(X_test)
    y_pred_probs = softmax(y_pred_logits)  # Apply softmax
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate per-class metrics
    class_accuracies = {}
    for i in range(config.NUM_CLASSES):
        if i in label_map:
            mask = y_test == i
            if np.sum(mask) > 0:  # Avoid division by zero
                class_acc = np.mean(y_pred[mask] == i)
                class_accuracies[label_map[i]] = class_acc
                print(f"  {label_map[i]}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    # Calculate emotion fairness
    emotion_fairness = calculate_emotion_fairness(class_accuracies)
    print(f"Emotion Fairness Score (min/max ratio): {emotion_fairness:.4f}")
    
    # Return results dictionary
    return {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'class_accuracies': class_accuracies,
        'emotion_fairness': emotion_fairness,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }