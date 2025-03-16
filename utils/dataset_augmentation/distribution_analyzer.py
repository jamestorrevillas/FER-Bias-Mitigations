# utils/dataset_augmentation/distribution_analyzer.py

"""
Analyzes dataset distributions and determines optimal augmentation strategies.
"""

import numpy as np
import pandas as pd
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns
from . import config_settings

def analyze_emotion_distribution(labels):
    """
    Analyze emotion class distribution and calculate statistics.
    
    Args:
        labels: Array of emotion labels
        
    Returns:
        Dictionary with distribution statistics
    """
    # Count frequency of each label
    label_counts = np.bincount(labels, minlength=max(labels)+1)
    
    # Skip index 0 if it's not used (for RAF-DB starting at 1)
    if label_counts[0] == 0 and len(label_counts) > 1:
        label_counts = label_counts[1:]
    
    total_samples = np.sum(label_counts)
    max_count = np.max(label_counts)
    min_count = np.min(label_counts[label_counts > 0])  # Only consider classes that exist
    
    # Calculate imbalance metrics
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return {
        'counts': label_counts,
        'total': total_samples,
        'max_count': max_count,
        'min_count': min_count,
        'imbalance_ratio': imbalance_ratio,
        'percentages': (label_counts / total_samples * 100) if total_samples > 0 else np.zeros_like(label_counts)
    }

def analyze_demographic_distribution(demographic_info):
    """
    Analyze demographic distribution (gender, age) and calculate statistics.
    
    Args:
        demographic_info: List of dictionaries with demographic information
        
    Returns:
        Dictionary with gender and age distribution statistics
    """
    # Convert to pandas DataFrame for easier analysis
    demo_df = pd.DataFrame(demographic_info)
    
    # Get gender distribution
    gender_counts = demo_df['gender'].value_counts().sort_index()
    total_gender = gender_counts.sum()
    max_gender = gender_counts.max()
    min_gender = gender_counts.min()
    gender_imbalance = max_gender / min_gender if min_gender > 0 else float('inf')
    
    # Get age distribution
    age_counts = demo_df['age'].value_counts().sort_index()
    total_age = age_counts.sum()
    max_age = age_counts.max()
    min_age = age_counts.min()
    age_imbalance = max_age / min_age if min_age > 0 else float('inf')
    
    return {
        'gender': {
            'counts': gender_counts.to_dict(),
            'total': total_gender,
            'max_count': max_gender,
            'min_count': min_gender,
            'imbalance_ratio': gender_imbalance,
            'percentages': {k: v/total_gender*100 for k, v in gender_counts.to_dict().items()} if total_gender > 0 else {}
        },
        'age': {
            'counts': age_counts.to_dict(),
            'total': total_age,
            'max_count': max_age,
            'min_count': min_age,
            'imbalance_ratio': age_imbalance,
            'percentages': {k: v/total_age*100 for k, v in age_counts.to_dict().items()} if total_age > 0 else {}
        }
    }

def analyze_intersectional_distribution(labels, demographic_info):
    """
    Analyze intersectional categories (emotion+demographic combinations).
    
    Args:
        labels: Array of emotion labels
        demographic_info: List of dictionaries with demographic information
        
    Returns:
        Dictionary with intersectional distribution statistics
    """
    intersections = {}
    
    # Convert demographic_info to DataFrame if it's not already
    if not isinstance(demographic_info, pd.DataFrame):
        demo_df = pd.DataFrame(demographic_info)
    else:
        demo_df = demographic_info
    
    # Add emotion to DataFrame
    demo_df['emotion'] = labels
    
    # Create emotion+gender intersections
    emotion_gender_cross = pd.crosstab(demo_df['emotion'], demo_df['gender'])
    
    # Create emotion+age intersections
    emotion_age_cross = pd.crosstab(demo_df['emotion'], demo_df['age'])
    
    # Find critical intersections (below threshold)
    critical_intersections = []
    
    # Check emotion+gender for critical intersections
    for emotion, gender_counts in emotion_gender_cross.iterrows():
        for gender, count in gender_counts.items():
            key = f"emotion_{emotion}_gender_{gender}"
            if count < config_settings.CRITICAL_INTERSECTION_THRESHOLD:
                critical_intersections.append((key, count))
    
    # Check emotion+age for critical intersections
    for emotion, age_counts in emotion_age_cross.iterrows():
        for age, count in age_counts.items():
            key = f"emotion_{emotion}_age_{age}"
            if count < config_settings.CRITICAL_INTERSECTION_THRESHOLD:
                critical_intersections.append((key, count))
    
    # Sort critical intersections by count (ascending)
    critical_intersections.sort(key=lambda x: x[1])
    
    return {
        'emotion_gender': emotion_gender_cross.to_dict(),
        'emotion_age': emotion_age_cross.to_dict(),
        'critical_intersections': dict(critical_intersections)
    }

def calculate_optimal_multipliers(distribution, target_imbalance=None, max_factor=None):
    """
    Calculate optimal multipliers to achieve target balance.
    
    Args:
        distribution: Dictionary with counts or array of counts
        target_imbalance: Target maximum imbalance ratio (default from config)
        max_factor: Maximum augmentation factor allowed (default from config)
        
    Returns:
        Dictionary mapping class indices to multipliers
    """
    # Use defaults from config if not specified
    if target_imbalance is None:
        target_imbalance = config_settings.TARGET_IMBALANCE_RATIO
    
    if max_factor is None:
        max_factor = config_settings.MAX_AUGMENTATION_FACTOR
    
    # Extract counts from distribution
    if isinstance(distribution, dict) and 'counts' in distribution:
        counts = distribution['counts']
    else:
        counts = distribution
    
    # Convert to array if it's a dict
    if isinstance(counts, dict):
        keys = sorted(counts.keys())
        counts_array = np.array([counts[k] for k in keys])
    else:
        counts_array = np.array(counts)
        keys = np.arange(len(counts_array))
    
    # Find max count for reference
    max_count = np.max(counts_array)
    
    # Calculate target minimum count
    target_min_count = max_count / target_imbalance
    
    # Calculate multipliers
    multipliers = {}
    for i, count in zip(keys, counts_array):
        if count > 0:  # Only calculate for classes that exist
            if count < target_min_count:
                # Need augmentation to reach target
                multiplier = min(target_min_count / count, max_factor)
            else:
                # No augmentation needed
                multiplier = 1.0
            
            multipliers[i] = round(multiplier, 1)
    
    return multipliers

def generate_emotion_plan(emotion_distribution):
    """
    Generate a comprehensive plan for emotion-based augmentation.
    
    Args:
        emotion_distribution: Distribution statistics from analyze_emotion_distribution
        
    Returns:
        Dictionary with augmentation plan
    """
    # Calculate optimal multipliers
    multipliers = calculate_optimal_multipliers(emotion_distribution)
    
    # Get counts
    counts = emotion_distribution['counts']
    
    # Create augmentation plan
    augmentation_plan = {}
    total_augmentations = 0
    
    for emotion, multiplier in multipliers.items():
        # Calculate how many augmentations to generate
        count = counts[emotion] if isinstance(counts, dict) else counts[int(emotion)]
        augmentations_needed = int(count * (multiplier - 1))
        
        augmentation_plan[emotion] = {
            'emotion': emotion,
            'current_count': count,
            'target_count': int(count * multiplier),
            'augmentations_needed': augmentations_needed,
            'augmentation_factor': multiplier
        }
        
        total_augmentations += augmentations_needed
    
    # Add summary info
    augmentation_plan['summary'] = {
        'total_original': emotion_distribution['total'],
        'total_augmentations': total_augmentations,
        'projected_total': emotion_distribution['total'] + total_augmentations,
        'current_imbalance': emotion_distribution['imbalance_ratio'],
        'target_imbalance': config_settings.TARGET_IMBALANCE_RATIO
    }
    
    return augmentation_plan

def generate_demographic_plan(demographic_distribution):
    """
    Generate a comprehensive plan for demographic-based augmentation.
    
    Args:
        demographic_distribution: Distribution from analyze_demographic_distribution
        
    Returns:
        Dictionary with augmentation plan
    """
    # Calculate optimal multipliers for gender
    gender_multipliers = calculate_optimal_multipliers(demographic_distribution['gender'])
    
    # Calculate optimal multipliers for age
    age_multipliers = calculate_optimal_multipliers(demographic_distribution['age'])
    
    return {
        'gender': gender_multipliers,
        'age': age_multipliers
    }

def generate_intersectional_plan(emotion_dist, demographic_dist, intersection_dist):
    """
    Generate a plan that addresses intersectional imbalances.
    
    Args:
        emotion_dist: Distribution from analyze_emotion_distribution
        demographic_dist: Distribution from analyze_demographic_distribution
        intersection_dist: Distribution from analyze_intersectional_distribution
        
    Returns:
        Dictionary with integrated augmentation plan
    """
    # Start with basic emotion plan
    base_plan = generate_emotion_plan(emotion_dist)
    
    # Add demographic multipliers
    demo_plan = generate_demographic_plan(demographic_dist)
    
    # Create dictionary for intersection boosts
    intersection_boosts = {}
    
    # Add boosts for critical intersections
    for key, count in intersection_dist['critical_intersections'].items():
        # The smaller the count, the higher the boost
        severity = config_settings.CRITICAL_INTERSECTION_THRESHOLD / max(count, 1)
        boost = min(severity, config_settings.CRITICAL_INTERSECTION_BOOST)
        intersection_boosts[key] = boost
    
    # Integrate everything into a master plan
    master_plan = {
        'base_emotion_plan': base_plan,
        'demographic_multipliers': demo_plan,
        'intersection_boosts': intersection_boosts,
        'summary': base_plan['summary']
    }
    
    return master_plan

def print_augmentation_plan(plan):
    """
    Print a human-readable summary of the augmentation plan.
    
    Args:
        plan: Augmentation plan dictionary
    """
    if 'summary' in plan:
        print("\nAugmentation Plan:")
        print("-" * 80)
        print(f"{'Category':<10} {'Current':<10} {'Target':<10} {'To Generate':<15} {'Factor':<10}")
        print("-" * 80)
        
        for key, details in plan.items():
            if key != 'summary' and isinstance(details, dict) and 'current_count' in details:
                print(f"{details.get('emotion', key):<10} {details['current_count']:<10} "
                      f"{details['target_count']:<10} {details['augmentations_needed']:<15} "
                      f"{details['augmentation_factor']:.2f}")
        
        print("-" * 80)
        summary = plan['summary']
        print(f"Total augmentations to generate: {summary['total_augmentations']:,}")
        print(f"Current dataset size: {summary['total_original']:,}")
        print(f"Projected final size: {summary['projected_total']:,}")
        print(f"Current imbalance ratio: {summary['current_imbalance']:.2f}:1")
        print(f"Target imbalance ratio: {summary['target_imbalance']:.2f}:1")
        print("-" * 80)
    else:
        print("Plan does not contain summary information.")