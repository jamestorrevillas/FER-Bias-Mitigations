# diagnose_augmented_data.py

import os
import numpy as np
import matplotlib.pyplot as plt

def diagnose_dataset(dataset_path, output_dir="diagnostic_images"):
    """
    Diagnose issues with an augmented dataset
    
    Args:
        dataset_path: Path to .npy file containing augmented images
        output_dir: Directory to save diagnostic images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}...")
    try:
        # Load the augmented images
        images = np.load(dataset_path)
        
        # Print basic information
        print(f"Successfully loaded {len(images)} images")
        print(f"Shape: {images.shape}")
        print(f"Data type: {images.dtype}")
        print(f"Value range: [{np.min(images):.4f}, {np.max(images):.4f}]")
        print(f"Mean: {np.mean(images):.4f}, Std: {np.std(images):.4f}")
        
        # Check for NaN or infinite values
        has_nan = np.isnan(images).any()
        has_inf = np.isinf(images).any()
        print(f"Contains NaN values: {has_nan}")
        print(f"Contains infinite values: {has_inf}")
        
        # Show histogram of pixel values
        plt.figure(figsize=(10, 6))
        plt.hist(images.flatten(), bins=50)
        plt.title('Distribution of Pixel Values')
        plt.savefig(os.path.join(output_dir, "pixel_distribution.png"))
        plt.close()
        
        # Save sample images with proper normalization
        for i in range(min(10, len(images))):
            img = images[i].squeeze()
            
            # Create a figure with both original scaling and [0,1] scaling
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original scaling
            im1 = ax1.imshow(img, cmap='gray')
            ax1.set_title(f"Original Range: [{np.min(img):.2f}, {np.max(img):.2f}]")
            plt.colorbar(im1, ax=ax1)
            ax1.axis('off')
            
            # Normalized to [0,1] for viewing
            if np.min(img) < 0 or np.max(img) > 1:
                # Convert to [0,1] range for display
                if np.min(img) >= -1 and np.max(img) <= 1:
                    # Convert from [-1,1] to [0,1]
                    norm_img = (img + 1) / 2
                else:
                    # Min-max normalization
                    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
            else:
                norm_img = img
                
            im2 = ax2.imshow(norm_img, cmap='gray')
            ax2.set_title(f"Normalized to [0,1] for display")
            plt.colorbar(im2, ax=ax2)
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_image_{i}.png"))
            plt.close()
            
        print(f"Diagnostic images saved to {output_dir}")
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    # FER+ augmented dataset
    ferplus_path = "resources/dataset/fer/augmented/augmented_images.npy"
    if os.path.exists(ferplus_path):
        print("Diagnosing FER+ augmented dataset:")
        diagnose_dataset(ferplus_path, "ferplus_diagnostics")
    
    # RAF-DB augmented dataset
    rafdb_path = "resources/dataset/raf-db/augmented/augmented_images.npy"
    if os.path.exists(rafdb_path):
        print("\nDiagnosing RAF-DB augmented dataset:")
        diagnose_dataset(rafdb_path, "rafdb_diagnostics")