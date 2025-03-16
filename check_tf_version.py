# check_tf_version.py
# Run this script in your VSCode terminal with: python check_tf_version.py

import os
import sys

# Print Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Try to import TensorFlow and get its version
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    # Check if GPU is available
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Additional TensorFlow details
    print("\nTensorFlow build information:")
    print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"  Built with GPU support: {tf.test.is_built_with_gpu_support()}")
    
except ImportError:
    print("TensorFlow is not installed.")
    
# Try to check model version and format
model_path = "resources/models/baseline-ferplus-model.h5"
if os.path.exists(model_path):
    try:
        # Check if h5py is available to examine the H5 file
        import h5py
        with h5py.File(model_path, 'r') as f:
            # Try to get Keras version from the model file
            if 'keras_version' in f.attrs:
                print(f"\nModel was created with Keras version: {f.attrs['keras_version'].decode('utf-8')}")
            # Get more details about model architecture
            if 'model_config' in f.attrs:
                print("Model config is available in the file")
                
                # Try to identify TensorFlow version from architecture patterns
                config_str = f.attrs['model_config'].decode('utf-8')
                if '"groups": 1' in config_str and '"SeparableConv2D"' in config_str:
                    print("Model appears to use newer TensorFlow 2.x constructs (like 'groups' parameter)")
                if '"depthwise_constraint"' in config_str and '"pointwise_constraint"' in config_str:
                    print("Model uses SeparableConv2D with depthwise/pointwise constraints (TF 2.x)")
    except ImportError:
        print("h5py is not installed. Cannot check model details.")
    except Exception as e:
        print(f"Error examining model file: {str(e)}")
else:
    print(f"Model file not found at {model_path}")

# Check for any other relevant TensorFlow dependencies
print("\nChecking for other relevant packages:")
packages = ["numpy", "pandas", "scikit-learn", "matplotlib", "seaborn"]
for package in packages:
    try:
        module = __import__(package)
        print(f"  {package}: {module.__version__}")
    except ImportError:
        print(f"  {package}: Not installed")
    except AttributeError:
        print(f"  {package}: Installed (version unknown)")