# bias_mitigation_approaches/dynamic_weighting/utils/weight_scheduler.py

import numpy as np
from collections import defaultdict
from . import config
from . import metrics

class DynamicWeightScheduler:
    """
    Dynamic sample weight scheduler that adjusts weights based on fairness feedback
    """
    
    def __init__(self, initial_weights=None, emotion_labels=None, demographic_info=None):
        """
        Initialize the weight scheduler
        
        Args:
            initial_weights: Initial sample weights (default: uniform weights)
            emotion_labels: Emotion labels corresponding to the weights
            demographic_info: Demographic information for each sample
        """
        self.weights = initial_weights
        self.emotion_labels = emotion_labels
        self.demographic_info = demographic_info
        self.emotion_weight_multipliers = np.ones(config.NUM_CLASSES)
        self.gender_weight_multipliers = {gender: 1.0 for gender in config.GENDER_LABELS.values()}
        self.age_weight_multipliers = {age: 1.0 for age in config.AGE_GROUPS.values()}
        self.intersection_multipliers = {}
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
        
        # 2. Update demographic weight multipliers
        for gender, gap in gender_gaps.items():
            # Apply multiplicative factor based on performance gap
            multiplier = 1.0 + gap * config.INITIAL_WEIGHT_MULTIPLIER
            multiplier = min(multiplier, config.MAX_WEIGHT_MULTIPLIER)
            
            # Apply weight decay for stability
            if self.iteration > 1:
                prev_multiplier = self.gender_weight_multipliers[gender]
                multiplier = prev_multiplier * (1 - config.WEIGHT_DECAY_FACTOR) + multiplier * config.WEIGHT_DECAY_FACTOR
            
            self.gender_weight_multipliers[gender] = multiplier
        
        # Do the same for age groups
        for age, gap in age_gaps.items():
            multiplier = 1.0 + gap * config.INITIAL_WEIGHT_MULTIPLIER
            multiplier = min(multiplier, config.MAX_WEIGHT_MULTIPLIER)
            
            if self.iteration > 1:
                prev_multiplier = self.age_weight_multipliers[age]
                multiplier = prev_multiplier * (1 - config.WEIGHT_DECAY_FACTOR) + multiplier * config.WEIGHT_DECAY_FACTOR
            
            self.age_weight_multipliers[age] = multiplier
        
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
        
        # 4. Update intersection-based weights
        self.intersection_multipliers = {}
        
        # Process gender-emotion intersections
        for demo, emotion, gap in metrics.identify_underperforming_intersections(gender_metrics):
            key = f"gender_{demo}_emotion_{emotion}"
            boost_factor = 1.0 + min(gap * config.INTERSECTION_WEIGHT_MULTIPLIER, 
                                    config.MAX_INTERSECTION_MULTIPLIER - 1.0)
            self.intersection_multipliers[key] = boost_factor
        
        # Process age-emotion intersections
        for demo, emotion, gap in metrics.identify_underperforming_intersections(age_metrics):
            key = f"age_{demo}_emotion_{emotion}"
            boost_factor = 1.0 + min(gap * config.INTERSECTION_WEIGHT_MULTIPLIER, 
                                    config.MAX_INTERSECTION_MULTIPLIER - 1.0)
            self.intersection_multipliers[key] = boost_factor
        
        # 5. Apply updated multipliers to individual sample weights
        if self.weights is not None and self.emotion_labels is not None:
            # Reset weights
            self.weights = np.ones(len(self.emotion_labels))
            
            # Apply emotion-based weights
            for i, label in enumerate(self.emotion_labels):
                self.weights[i] *= self.emotion_weight_multipliers[label]
                
                # Apply intersectional weights if demographic information is available
                if self.demographic_info is not None and i < len(self.demographic_info):
                    demo_info = self.demographic_info[i]
                    
                    # Get RAF-DB emotion from FER+ emotion
                    raf_emotion = None
                    if label in config.FERPLUS_TO_RAFDB and config.FERPLUS_TO_RAFDB[label] is not None:
                        raf_emotion = config.RAFDB_EMOTIONS[config.FERPLUS_TO_RAFDB[label]]
                    
                    if raf_emotion:
                        # Apply gender-emotion intersection weight if available
                        gender_val = demo_info.get('gender')
                        if gender_val is not None and gender_val in config.GENDER_LABELS:
                            gender_name = config.GENDER_LABELS[gender_val]
                            gender_key = f"gender_{gender_name}_emotion_{raf_emotion}"
                            if gender_key in self.intersection_multipliers:
                                self.weights[i] *= self.intersection_multipliers[gender_key]
                        
                        # Apply age-emotion intersection weight if available
                        age_val = demo_info.get('age')
                        if age_val is not None and age_val in config.AGE_GROUPS:
                            age_name = config.AGE_GROUPS[age_val]
                            age_key = f"age_{age_name}_emotion_{raf_emotion}"
                            if age_key in self.intersection_multipliers:
                                self.weights[i] *= self.intersection_multipliers[age_key]
            
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
            'age_multipliers': self.age_weight_multipliers,
            'intersection_multipliers': self.intersection_multipliers
        }