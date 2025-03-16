# utils/dataset_augmentation/image_processor.py

"""
Generates augmented images based on augmentation plans and creates
visualizations of distributions and results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import config_settings

def random_crop(img, crop_size=96):
    """
    Random crop an image to the specified size
    
    Args:
        img: Input image (2D or 3D)
        crop_size: Size to crop to (square)
        
    Returns:
        Cropped image
    """
    # Handle dimensions properly
    if len(img.shape) == 3:
        height, width = img.shape[0], img.shape[1]
        has_channels = True
    else:
        height, width = img.shape
        has_channels = False
    
    # Calculate valid crop ranges
    dy = height - crop_size
    dx = width - crop_size
    
    # Select random position
    y = np.random.randint(0, max(1, dy + 1))
    x = np.random.randint(0, max(1, dx + 1))
    
    # Return cropped image with correct dimensions
    if has_channels:
        return img[y:y+crop_size, x:x+crop_size, :]
    else:
        return img[y:y+crop_size, x:x+crop_size]

def apply_histogram_equalization(img):
    """
    Apply histogram equalization to increase global contrast
    
    Args:
        img: Input image
        
    Returns:
        Contrast-enhanced image
    """
    # Ensure image is in uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    # Apply to grayscale image
    if len(img.shape) == 2 or img.shape[2] == 1:
        equalized = cv2.equalizeHist(img.squeeze())
        return equalized.reshape(img.shape) if len(img.shape) > 2 else equalized
    
    return img

def create_augmentation_generator(settings=None):
    """
    Create a configured augmentation generator using paper's approach.
    
    Args:
        settings: Dictionary of generator settings (defaults from config if None)
        
    Returns:
        Configured ImageDataGenerator
    """
    if settings is None:
        settings = config_settings.DEFAULT_AUGMENTATION_SETTINGS
    
    # Extract custom parameters
    generator_settings = settings.copy()
    random_crop_enabled = generator_settings.pop('random_crop', False)
    crop_size = generator_settings.pop('crop_size', 96)
    
    # Define preprocessing function using paper's weighted summation approach
    def custom_preprocessing(image):
        """Custom preprocessing function for augmentation"""
        # Save original shape and data type
        original_shape = image.shape
        original_dtype = image.dtype
        has_channel_dim = len(original_shape) > 2
        
        # Convert to uint8 for processing (OpenCV expects uint8)
        if original_dtype != np.uint8:
            if np.max(image) <= 1.0:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image.astype(np.uint8)
        else:
            image_uint8 = image.copy()
        
        # Create copies for each strategy
        # For 2D/3D compatibility
        if has_channel_dim:
            image_2d = image_uint8.squeeze()
        else:
            image_2d = image_uint8
        
        # Strategy One: Geometric transformations
        strategy_one_result = image_2d.copy()
        
        # Apply cropping if enabled and image is large enough
        if random_crop_enabled and \
           strategy_one_result.shape[0] > crop_size and \
           strategy_one_result.shape[1] > crop_size:
            
            strategy_one_result = random_crop(
                strategy_one_result, 
                crop_size
            )
        
        # Strategy Two: Histogram equalization
        strategy_two_result = cv2.equalizeHist(image_2d.copy())
        
        # Make sure both have the same shape for weighted addition
        if strategy_one_result.shape != strategy_two_result.shape:
            # Resize to match strategy one result
            strategy_two_result = cv2.resize(
                strategy_two_result, 
                (strategy_one_result.shape[1], strategy_one_result.shape[0])
            )
        
        # Apply weighted summation
        result = cv2.addWeighted(strategy_one_result, 0.5, strategy_two_result, 0.5, 0)
        
        # Convert back to original data type
        if original_dtype != np.uint8:
            if np.max(image) <= 1.0:
                result = result.astype(np.float32) / 255.0
            else:
                result = result.astype(original_dtype)
        
        # Restore original dimensions if needed
        if has_channel_dim:
            # Add channel dimension back
            if len(result.shape) == 2:
                result = result.reshape(result.shape + (1,))
        
        return result
    
    # Add custom preprocessing to generator settings
    generator_settings['preprocessing_function'] = custom_preprocessing
    
    return ImageDataGenerator(**generator_settings)

def generate_augmentations(images, labels, augmentation_plan, metadata=None):
    """
    Generate augmented images based on the augmentation plan.
    
    Args:
        images: Array of original images
        labels: Array of labels
        augmentation_plan: Plan from distribution_analyzer
        metadata: Additional information (e.g., votes, demographic info)
        
    Returns:
        Tuple of (augmented_images, augmented_labels, original_indices, augmented_metadata)
    """
    datagen = create_augmentation_generator()
    augmented_images = []
    augmented_labels = []
    original_indices = []
    augmented_metadata = []
    
    # Initialize tracking
    augmentation_counts = {}
    
    # Group images by label
    label_indices = {}
    for i, label in enumerate(labels):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(i)
    
    # Check if we're dealing with emotion-only or intersectional plan
    is_intersectional = 'base_emotion_plan' in augmentation_plan
    
    # Extract the emotion plan
    if is_intersectional:
        emotion_plan = augmentation_plan['base_emotion_plan']
        demographic_multipliers = augmentation_plan['demographic_multipliers']
        intersection_boosts = augmentation_plan['intersection_boosts']
    else:
        emotion_plan = augmentation_plan
    
    # Create augmentations based on emotion plan
    for emotion_idx, plan_details in tqdm(emotion_plan.items(), desc="Processing categories"):
        if emotion_idx == 'summary':
            continue
            
        if 'augmentations_needed' not in plan_details:
            continue
            
        if plan_details['augmentations_needed'] <= 0:
            print(f"  Category {emotion_idx}: No augmentation needed")
            continue
            
        print(f"  Category {emotion_idx}: Generating {plan_details['augmentations_needed']} augmentations")
        
        # Get indices of all images with this category
        indices = label_indices.get(int(emotion_idx), [])
        if not indices:
            print(f"  Warning: No images found for category {emotion_idx}")
            continue
        
        # Calculate augmentations per image
        augmentations_per_image = max(1, int(plan_details['augmentations_needed'] / len(indices)))
        remainder = plan_details['augmentations_needed'] % len(indices)
        
        # Generate augmentations for each image
        for i, idx in enumerate(indices):
            img = images[idx]
            num_augs = augmentations_per_image + (1 if i < remainder else 0)
            
            # Apply demographic and intersection boosts if applicable
            if is_intersectional and isinstance(metadata, list) and idx < len(metadata):
                # Get demographic info for this image
                demo = metadata[idx]
                
                # Apply gender boost if applicable
                gender = demo.get('gender', None)
                gender_multiplier = demographic_multipliers['gender'].get(gender, 1.0) if gender is not None else 1.0
                
                # Apply age boost if applicable
                age = demo.get('age', None)
                age_multiplier = demographic_multipliers['age'].get(age, 1.0) if age is not None else 1.0
                
                # Apply intersection boost if applicable
                emotion_gender_key = f"emotion_{emotion_idx}_gender_{gender}" if gender is not None else None
                emotion_age_key = f"emotion_{emotion_idx}_age_{age}" if age is not None else None
                
                intersection_multiplier = 1.0
                if emotion_gender_key and emotion_gender_key in intersection_boosts:
                    intersection_multiplier *= intersection_boosts[emotion_gender_key]
                if emotion_age_key and emotion_age_key in intersection_boosts:
                    intersection_multiplier *= intersection_boosts[emotion_age_key]
                
                # Calculate total boost
                total_boost = gender_multiplier * age_multiplier * intersection_multiplier
                num_augs = int(num_augs * total_boost)
            
            # Apply quality-aware boost if metadata contains emotion_votes
            if isinstance(metadata, np.ndarray) and metadata.shape[0] > idx:
                votes = metadata[idx]
                sum_votes = np.sum(votes)
                if sum_votes > 0:
                    agreement = max(votes) / sum_votes
                    if agreement > config_settings.VERY_HIGH_AGREEMENT_THRESHOLD:
                        num_augs = int(num_augs * config_settings.VERY_HIGH_AGREEMENT_BOOST)
                    elif agreement > config_settings.HIGH_AGREEMENT_THRESHOLD:
                        num_augs = int(num_augs * config_settings.HIGH_AGREEMENT_BOOST)
            
            # Generate the augmentations
            img_reshaped = img.reshape((1,) + img.shape)
            
            # Debug print for first image
            if i == 0 and len(augmented_images) == 0:
                print(f"  Debug: Original image shape: {img.shape}, dtype: {img.dtype}")
                print(f"  Debug: Original image min: {np.min(img)}, max: {np.max(img)}")
            
            for _ in range(num_augs):
                aug_iter = datagen.flow(img_reshaped, batch_size=1)
                aug_img = next(aug_iter)[0]
                
                # Debug print for first augmented image
                if i == 0 and len(augmented_images) == 0:
                    print(f"  Debug: Augmented image shape: {aug_img.shape}, dtype: {aug_img.dtype}")
                    print(f"  Debug: Augmented image min: {np.min(aug_img)}, max: {np.max(aug_img)}")
                
                augmented_images.append(aug_img)
                augmented_labels.append(int(emotion_idx))
                original_indices.append(idx)
                
                # Copy metadata if available
                if isinstance(metadata, list) and idx < len(metadata):
                    augmented_metadata.append(metadata[idx])
                
                # Update tracking counts
                if emotion_idx not in augmentation_counts:
                    augmentation_counts[emotion_idx] = 0
                augmentation_counts[emotion_idx] += 1
    
    # Create augmentation statistics plot
    augmented_images_array = np.array(augmented_images) if augmented_images else np.array([])
    augmented_labels_array = np.array(augmented_labels) if augmented_labels else np.array([])
    original_indices_array = np.array(original_indices) if original_indices else np.array([])
    
    # Print summary stats about augmented images
    if len(augmented_images_array) > 0:
        print(f"Generated {len(augmented_images_array)} augmented images")
        print(f"Augmented images shape: {augmented_images_array.shape}")
        print(f"Augmented images dtype: {augmented_images_array.dtype}")
        print(f"Augmented images value range: [{np.min(augmented_images_array)}, {np.max(augmented_images_array)}]")
    
    return (augmented_images_array, 
            augmented_labels_array, 
            original_indices_array, 
            augmented_metadata)

def apply_custom_normalization(images):
    """
    Apply standardized normalization to images.
    
    Args:
        images: Array of images to normalize
        
    Returns:
        Array of normalized images
    """
    print("Applying normalization...")
    
    # Check the range of pixel values
    min_val = np.min(images)
    max_val = np.max(images)
    print(f"Image value range before normalization: [{min_val:.4f}, {max_val:.4f}]")
    
    if np.isnan(min_val) or np.isnan(max_val) or np.isinf(min_val) or np.isinf(max_val):
        print("WARNING: Dataset contains NaN or Inf values!")
    
    # Standardized normalization approach to ensure consistency
    normalized_images = []
    
    for img in tqdm(images, desc="Normalizing"):
        # First ensure all values are in [0, 1] range
        if img.dtype == np.uint8:
            # Convert from [0, 255] to [0, 1]
            img_float = img.astype(np.float32) / 255.0
        elif np.max(img) > 1.0 or img.dtype != np.float32:
            # If image has values > 1.0, assume it's in [0, 255] range
            img_float = img.astype(np.float32) / 255.0
        else:
            # Already in [0, 1] range
            img_float = img.astype(np.float32)
        
        # Apply the final normalization to [-1, 1] range
        img_norm = (img_float - 0.5) * 2.0
        normalized_images.append(img_norm)
    
    result = np.array(normalized_images)
    print(f"Range after normalization: [{np.min(result):.4f}, {np.max(result):.4f}]")
    return result

def custom_normalize_image(image):
    """Normalize a single image."""
    # Convert to float32 for processing
    image = image.astype('float32')
    
    # Handle different input ranges
    if image.dtype == np.uint8 or np.max(image) > 1.0:
        # Scale from [0, 255] to [0, 1]
        image = image / 255.0
    
    # Convert from [0,1] to [-1,1] range
    image = (image - 0.5) * 2.0
    return image

def plot_distribution(distribution, title, save_path, x_labels=None):
    """
    Plot frequency distribution with statistics.
    
    Args:
        distribution: Dictionary or array of counts
        title: Plot title
        save_path: Path to save the visualization
        x_labels: Optional labels for x-axis categories
    """
    plt.figure(figsize=(12, 6))
    
    # Extract counts
    if isinstance(distribution, dict) and 'counts' in distribution:
        counts = distribution['counts']
    else:
        counts = distribution
    
    # Convert dict to arrays if needed
    if isinstance(counts, dict):
        keys = sorted(counts.keys())
        values = [counts[k] for k in keys]
        if x_labels is None and len(keys) <= len(config_settings.RAFDB_EMOTIONS):
            # Try to use emotion labels if they match
            x_labels = [config_settings.RAFDB_EMOTIONS.get(k, k) for k in keys]
    else:
        values = counts
        keys = range(len(values))
        if x_labels is None and len(keys) <= len(config_settings.RAFDB_EMOTIONS):
            x_labels = [config_settings.RAFDB_EMOTIONS.get(i, i) for i in keys]
    
    # Default x-labels if none provided
    if x_labels is None:
        x_labels = [str(k) for k in keys]
    
    # Plot
    bars = plt.bar(x_labels, values)
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height):,}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save and close
    plt.savefig(save_path)
    plt.close()

def plot_augmentation_plan(original_dist, augmentation_plan, save_path):
    """
    Visualize augmentation plan and projected distribution.
    
    Args:
        original_dist: Original distribution statistics
        augmentation_plan: Augmentation plan
        save_path: Path to save the visualization
    """
    # Create directory for save path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Extract emotion plan if it's an intersectional plan
    if 'base_emotion_plan' in augmentation_plan:
        emotion_plan = augmentation_plan['base_emotion_plan']
    else:
        emotion_plan = augmentation_plan
    
    # Skip if no summary
    if 'summary' not in emotion_plan:
        return
    
    # Extract original counts
    original_counts = original_dist['counts']
    if isinstance(original_counts, dict):
        keys = sorted(original_counts.keys())
        original_counts_array = np.array([original_counts[k] for k in keys])
    else:
        original_counts_array = np.array(original_counts)
        keys = range(len(original_counts_array))
    
    # Calculate projected counts
    projected_counts = []
    for key in keys:
        if key in emotion_plan and key != 'summary':
            projected_counts.append(emotion_plan[key]['target_count'])
        else:
            idx = int(key) if not isinstance(key, int) else key
            if idx < len(original_counts_array):
                projected_counts.append(original_counts_array[idx])
            else:
                projected_counts.append(0)
    
    # Convert to numpy array
    projected_counts = np.array(projected_counts)
    
    # Calculate percentages
    total_original = np.sum(original_counts_array)
    total_projected = np.sum(projected_counts)
    original_pct = original_counts_array / total_original * 100
    projected_pct = projected_counts / total_projected * 100
    
    # Get category labels (emotions in this case)
    if all(isinstance(k, int) for k in keys) and max(keys) < len(config_settings.RAFDB_EMOTIONS):
        category_labels = [config_settings.RAFDB_EMOTIONS.get(k, k) for k in keys]
    else:
        category_labels = [str(k) for k in keys]
    
    # Plot comparison as percentages
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(category_labels))
    width = 0.35
    
    # Original percentages
    rects1 = plt.bar(x - width/2, original_pct, width, label='Original Distribution')
    
    # Projected percentages after augmentation
    rects2 = plt.bar(x + width/2, projected_pct, width, label='Projected Distribution')
    
    plt.title('Distribution Comparison (% of dataset)')
    plt.xlabel('Category')
    plt.ylabel('Percentage')
    plt.xticks(x, category_labels, rotation=45, ha='right')
    plt.legend()
    
    # Add percentage labels
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(save_path), 'distribution_comparison_pct.png'))
    plt.close()
    
    # Plot comparison as raw counts
    plt.figure(figsize=(14, 8))
    
    # Original counts
    rects1 = plt.bar(x - width/2, original_counts_array, width, label='Original Count')
    
    # Projected counts after augmentation
    rects2 = plt.bar(x + width/2, projected_counts, width, label='Projected Count')
    
    plt.title('Count Comparison')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(x, category_labels, rotation=45, ha='right')
    plt.legend()
    
    # Add count labels
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 5,
                f'{int(height):,}', ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 5,
                f'{int(height):,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_samples(original_images, augmented_images, indices, save_path=None):
    """
    Create visualization of original vs augmented image pairs with proper normalization.
    
    Args:
        original_images: Array of original images
        augmented_images: Array of augmented images
        indices: Indices of original images used to generate the augmented images
        save_path: Path to save the visualization (optional)
    """
    num_samples = min(len(indices), 5)
    plt.figure(figsize=(12, 2*num_samples))
    
    for i, idx in enumerate(indices[:num_samples]):
        orig_idx = indices[i]
        
        # Get original image
        orig_img = original_images[orig_idx].squeeze()
        
        # Normalize for display if needed
        if np.min(orig_img) < 0 or np.max(orig_img) > 1:
            if np.min(orig_img) >= -1 and np.max(orig_img) <= 1:
                # Convert from [-1,1] to [0,1]
                orig_img_display = (orig_img + 1) / 2
            else:
                # Min-max normalization
                orig_img_display = (orig_img - np.min(orig_img)) / (np.max(orig_img) - np.min(orig_img))
        else:
            orig_img_display = orig_img
        
        # Original image
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(orig_img_display, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Original {orig_idx}\n[{np.min(orig_img):.2f}, {np.max(orig_img):.2f}]")
        plt.axis('off')
        
        # Get augmented image
        aug_img = augmented_images[i].squeeze()
        
        # Normalize for display if needed
        if np.min(aug_img) < 0 or np.max(aug_img) > 1:
            if np.min(aug_img) >= -1 and np.max(aug_img) <= 1:
                # Convert from [-1,1] to [0,1]
                aug_img_display = (aug_img + 1) / 2
            else:
                # Min-max normalization
                aug_img_display = (aug_img - np.min(aug_img)) / (np.max(aug_img) - np.min(aug_img))
        else:
            aug_img_display = aug_img
        
        # Augmented image
        plt.subplot(num_samples, 2, 2*i + 2)
        plt.imshow(aug_img_display, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Augmented {i}\n[{np.min(aug_img):.2f}, {np.max(aug_img):.2f}]")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()

def create_augmentation_stats_plot(augmentation_counts, categories=None, save_path=None):
    """
    Create and save plots of augmentation statistics.
    
    Args:
        augmentation_counts: Dictionary of augmentation counts
        categories: Optional category names (e.g., emotion labels)
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    # Extract keys and values
    keys = sorted(augmentation_counts.keys())
    values = [augmentation_counts[k] for k in keys]
    
    # Get category labels if available
    if categories:
        labels = [categories.get(k, k) for k in keys]
    else:
        # Try to use emotion labels if keys are integers
        if all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in keys):
            labels = [config_settings.RAFDB_EMOTIONS.get(int(k), k) for k in keys]
        else:
            labels = keys
    
    # Create bar chart
    bars = plt.bar(labels, values)
    plt.title('Augmentations by Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Augmentations')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height):,}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.close()