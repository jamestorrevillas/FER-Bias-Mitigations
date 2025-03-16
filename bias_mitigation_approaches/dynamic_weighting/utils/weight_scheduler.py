# bias_mitigation_approaches/dynamic_weighting/utils/weight_scheduler.py

import numpy as np
from collections import defaultdict
from . import config
from . import metrics

class DynamicWeightScheduler:
    """
    Dynamic sample weight scheduler that adjusts weights based on fairness feedback
    """
    
    def __init__(self, initial_weights=None, emotion_labels=None):
        """
        Initialize the weight scheduler
        
        Args:
            initial_weights: Initial sample weights (default: uniform weights)
            emotion_labels: Emotion labels corresponding to the weights
        """
        self.weights = initial_weights
        self.emotion_labels = emotion_labels
        self.emotion_weight_multipliers = np.ones(config.NUM_CLASSES)
        self.gender_weight_multipliers = {gender: 1.0 for gender in config.GENDER_LABELS.values()}
        self.age_weight_multipliers = {age: 1.0 for age in config.AGE_GROUPS.values()}
        self.iteration = 0
        
        # Set default weight to 1.0 if not provided
        if self.weights is None and emotion_labels is not None:
            self.weights = np.ones(len(emotion_labels))
    
    def update_weights(self, fairness_metrics):
        """
        Update sample weights based on fairness metrics
        
        Args:
            fairness_metrics: Output from metrics.evaluate_model_fairness
            
        Returns:
            Updated sample weights
        """
        self.iteration += 1
        
        # Extract demographic metrics
        gender_metrics = fairness_metrics['gender_metrics']
        age_metrics = fairness_metrics['age_metrics']
        emotion_accuracies = fairness_metrics['emotion_accuracies']
        
        # 1. Identify underperforming demographics
        gender_gaps = metrics.identify_underperforming_groups(gender_metrics)
        age_gaps = metrics.identify_underperforming_groups(age_metrics)
        
        # 2. Identify underperforming intersections
        gender_intersections = metrics.identify_underperforming_intersections(gender_metrics)
        age_intersections = metrics.identify_underperforming_intersections(age_metrics)
        
        # 3. Update emotion weight multipliers
        for emotion_name, accuracy in emotion_accuracies.items():
            # Find emotion index
            emotion_idx = None
            for idx, name in config.RAFDB_EMOTIONS.items():
                if name.lower() == emotion_name.lower():
                    for fer_idx, raf_idx in config.FERPLUS_TO_RAFDB.items():
                        if raf_idx == idx:
                            emotion_idx = fer_idx
                            break
            
            if emotion_idx is not None:
                # Calculate weight multiplier based on relative performance
                # Lower performance = higher weight multiplier
                relative_performance = accuracy / max(emotion_accuracies.values())
                multiplier = 1.0 + (1.0 - relative_performance) * config.INITIAL_WEIGHT_MULTIPLIER
                
                # Limit the maximum multiplier
                multiplier = min(multiplier, config.MAX_WEIGHT_MULTIPLIER)
                
                # Apply weight decay for stability
                if self.iteration > 1:
                    # Blend with previous multiplier
                    prev_multiplier = self.emotion_weight_multipliers[emotion_idx]
                    multiplier = prev_multiplier * (1 - config.WEIGHT_DECAY_FACTOR) + multiplier * config.WEIGHT_DECAY_FACTOR
                
                self.emotion_weight_multipliers[emotion_idx] = multiplier
        
        # 4. Apply updated multipliers to individual sample weights
        if self.weights is not None and self.emotion_labels is not None:
            # Reset weights
            self.weights = np.ones(len(self.emotion_labels))
            
            # Apply emotion-based weights
            for i, label in enumerate(self.emotion_labels):
                self.weights[i] *= self.emotion_weight_multipliers[label]
            
            # Normalize weights for stability
            if np.sum(self.weights) > 0:
                avg_weight = np.mean(self.weights)
                self.weights = self.weights / avg_weight
        
        return self.weights
    
    def get_current_weights(self):
        """
        Get the current sample weights
        
        Returns:
            Current sample weights
        """
        return self.weights
    
    def get_weight_multipliers(self):
        """
        Get the current weight multipliers
        
        Returns:
            Dictionary of current weight multipliers
        """
        return {
            'emotion_multipliers': {
                config.FERPLUS_EMOTIONS[i]: self.emotion_weight_multipliers[i]
                for i in range(len(self.emotion_weight_multipliers))
                if i in config.FERPLUS_EMOTIONS
            },
            'gender_multipliers': self.gender_weight_multipliers,
            'age_multipliers': self.age_weight_multipliers
        }