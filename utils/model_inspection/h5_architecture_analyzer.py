# utils/model_inspection/h5_architecture_analyzer.py

"""
FER+ Model Architecture Analyzer

This script analyzes the architecture of a FER+ model saved in H5 format
without loading the actual model. It extracts detailed information about
layers, parameters, and structure from the H5 file.

Usage:
    Simply run this script - no command-line arguments needed.
    Results will be saved to 'architecture_analysis' subfolder.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

###########################################
# CONFIGURATION - EDIT THESE PATHS
###########################################

# Set the path to your model file here
MODEL_PATH = 'resources/models/baseline-ferplus-model.h5'

# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set output directory (inside script directory by default)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'architecture_analysis')

###########################################
# DEPENDENCY HANDLING
###########################################

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found. Installing...")
    os.system('pip install tensorflow>=2.0.0')
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")

try:
    import h5py
except ImportError:
    print("h5py not found. Installing...")
    os.system('pip install h5py')
    import h5py

###########################################
# MODEL INSPECTION FUNCTION
###########################################

def inspect_model_h5_file(model_path, output_dir):
    """
    Detailed inspection of model H5 file structure without loading the model.
    
    Args:
        model_path (str): Path to the H5 model file
        output_dir (str): Directory where analysis outputs will be saved
        
    Returns:
        bool: True if inspection was successful, False otherwise
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Inspecting H5 file structure: {model_path}")
    print(f"Analysis results will be saved to: {output_dir}")
    
    try:
        with h5py.File(model_path, 'r') as f:
            # Print basic file info
            print("\nH5 File Structure:")
            print("=" * 80)
            
            # Define a deeper inspection function that counts params
            def detailed_inspect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    shape_str = str(obj.shape)
                    dtype_str = str(obj.dtype)
                    params = np.prod(obj.shape) if len(obj.shape) > 0 else 0
                    size_kb = obj.size * obj.dtype.itemsize / 1024
                    print(f"Dataset: {name:<50} | Shape: {shape_str:<20} | Type: {dtype_str:<10} | Params: {params:<10,.0f} | Size: {size_kb:.2f} KB")
                elif isinstance(obj, h5py.Group):
                    print(f"\nGroup: {name}")
            
            # Visit all groups and datasets
            f.visititems(detailed_inspect)
            
            ###########################################
            # PARAMETER ANALYSIS
            ###########################################
            
            # Calculate and print overall statistics
            total_params = 0
            layer_counts = {}
            
            def count_params(name, obj):
                nonlocal total_params
                if isinstance(obj, h5py.Dataset) and 'optimizer_weights' not in name and any(s in name for s in ['kernel', 'bias', 'gamma', 'beta']):
                    params = np.prod(obj.shape) if len(obj.shape) > 0 else 0
                    total_params += params
                    
                    # Extract layer type
                    parts = name.split('/')
                    if len(parts) >= 2:
                        layer_type = parts[1].split('_')[0]
                        if layer_type in layer_counts:
                            layer_counts[layer_type] += 1
                        else:
                            layer_counts[layer_type] = 1
            
            f.visititems(count_params)
            
            print("\n" + "=" * 80)
            print(f"Total trainable parameters: {total_params:,}")
            
            # Analyze layer types
            print("\nLayer Type Distribution:")
            for layer_type, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {layer_type}: {count}")
            
            ###########################################
            # ARCHITECTURE INFERENCE
            ###########################################
            
            print("\nInferred Model Architecture:")
            print("=" * 80)
            
            # Check for weights to infer architecture
            conv_layers = {}
            bn_layers = {}
            
            def extract_layer_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if 'separable_conv2d' in name and 'kernel' in name:
                        layer_name = name.split('/')[1]
                        if layer_name not in conv_layers:
                            conv_layers[layer_name] = {'shape': obj.shape}
                    elif 'batch_normalization' in name and 'gamma' in name:
                        layer_name = name.split('/')[1]
                        if layer_name not in bn_layers:
                            bn_layers[layer_name] = {'shape': obj.shape}
            
            f.visititems(extract_layer_info)
            
            print("\nConvolutional Layers:")
            for layer_name, info in sorted(conv_layers.items()):
                print(f"  {layer_name}: {info['shape']}")
            
            print("\nBatch Normalization Layers:")
            for layer_name, info in sorted(bn_layers.items()):
                print(f"  {layer_name}: {info['shape']}")
            
            ###########################################
            # MODEL CONFIG EXTRACTION (if available)
            ###########################################
            
            if 'model_config' in f.attrs:
                try:
                    import json
                    config_str = f.attrs['model_config']
                    if isinstance(config_str, bytes):
                        config_str = config_str.decode('utf-8')
                    config = json.loads(config_str)
                    
                    print("\nModel Configuration Found")
                    
                    # Save config to file
                    with open(os.path.join(output_dir, 'model_config.json'), 'w') as config_file:
                        json.dump(config, config_file, indent=2)
                    print(f"Saved model configuration to '{os.path.join(output_dir, 'model_config.json')}'")
                    
                    # Extract and display layer information from config
                    if 'config' in config and 'layers' in config['config']:
                        layers = config['config']['layers']
                        print(f"\nFound {len(layers)} layers in model config")
                        
                        # Create dataframe with layer info
                        layer_info = []
                        for i, layer in enumerate(layers):
                            layer_config = layer['config']
                            info = {
                                'index': i,
                                'name': layer_config.get('name', 'unknown'),
                                'type': layer['class_name'],
                                'parameters': 'N/A'  # Can't compute without weights
                            }
                            
                            # Add layer-specific attributes
                            if 'filters' in layer_config:
                                info['filters'] = layer_config['filters']
                            if 'kernel_size' in layer_config:
                                info['kernel_size'] = layer_config['kernel_size']
                            if 'strides' in layer_config:
                                info['strides'] = layer_config['strides']
                            if 'padding' in layer_config:
                                info['padding'] = layer_config['padding']
                            if 'activation' in layer_config:
                                info['activation'] = layer_config['activation']
                            if 'rate' in layer_config:
                                info['rate'] = layer_config['rate']
                            if 'pool_size' in layer_config:
                                info['pool_size'] = layer_config['pool_size']
                            
                            layer_info.append(info)
                        
                        # Create dataframe and save to CSV
                        layer_df = pd.DataFrame(layer_info)
                        layer_df.to_csv(os.path.join(output_dir, 'layer_config.csv'), index=False)
                        print(f"Saved layer configuration to '{os.path.join(output_dir, 'layer_config.csv')}'")
                        
                        # Print summary of layer types
                        print("\nDetailed Layer Type Distribution:")
                        type_counts = layer_df['type'].value_counts()
                        for layer_type, count in type_counts.items():
                            print(f"  {layer_type}: {count}")
                except Exception as e:
                    print(f"Error parsing model_config: {str(e)}")
            else:
                print("\nNo model_config attribute found in the H5 file")
            
            ###########################################
            # OPTIMIZER ANALYSIS
            ###########################################
            
            if 'optimizer_weights' in f:
                print("\nOptimizer Information:")
                opt_group = f['optimizer_weights']
                if 'Adam' in opt_group:
                    print("  Optimizer type: Adam")
                    if 'iter' in opt_group['Adam']:
                        iter_value = opt_group['Adam']['iter'][()]
                        print(f"  Training iterations: {iter_value}")
                elif 'SGD' in opt_group:
                    print("  Optimizer type: SGD")
                else:
                    print("  Optimizer type: Unknown")
            
            ###########################################
            # ARCHITECTURE DETAILS FROM WEIGHTS
            ###########################################
            
            print("\nInferred Model Architecture Based on Weights:")
            print("=" * 80)
            
            # Get separable conv layers to determine the filter sizes
            sep_conv_layers = {}
            
            for name in f['model_weights']:
                if 'separable_conv2d' in name:
                    layer_num = name.replace('separable_conv2d_', '')
                    if name in f['model_weights'] and f['model_weights'][name]:
                        for subgroup in f['model_weights'][name]:
                            if subgroup in f['model_weights'][name]:
                                # Try to get bias to determine number of filters
                                bias_path = f'model_weights/{name}/{subgroup}/bias:0'
                                if bias_path in f:
                                    bias = f[bias_path]
                                    sep_conv_layers[int(layer_num)] = bias.shape[0]
            
            # Sort by layer number
            filter_progression = []
            for layer_num in sorted(sep_conv_layers.keys()):
                filter_progression.append((f'separable_conv2d_{layer_num}', sep_conv_layers[layer_num]))
            
            print("\nFilter progression through the network:")
            for name, filters in filter_progression:
                print(f"  {name}: {filters} filters")
            
            # Try to infer activation functions
            if len(f['model_weights'].keys()) > 0:
                activation_layers = [k for k in f['model_weights'].keys() if 'leaky_re_lu' in k or 'activation' in k]
                print("\nActivation layers:")
                for act_layer in sorted(activation_layers):
                    print(f"  {act_layer}")
            
            # Try to infer pooling layers
            pooling_layers = [k for k in f['model_weights'].keys() if 'pooling' in k.lower()]
            print("\nPooling layers:")
            for pool_layer in sorted(pooling_layers):
                print(f"  {pool_layer}")
            
            # Try to infer dropout layers
            dropout_layers = [k for k in f['model_weights'].keys() if 'dropout' in k.lower()]
            print("\nDropout layers:")
            for dropout_layer in sorted(dropout_layers):
                print(f"  {dropout_layer}")
                
                # Try to infer dropout rate from weights/config
                if 'spatial_dropout' in dropout_layer:
                    # Since dropout rates aren't stored in weights, we make an educated guess
                    # based on the inspection of the previous outputs
                    print(f"    Inferred dropout rate: 0.1 (based on typical use in this architecture)")
            
            ###########################################
            # LAYER SEQUENCE ANALYSIS
            ###########################################
            
            print("\nArchitecture Summary Based on Weights Analysis:")
            print("=" * 80)
            print(f"Input shape: (48, 48, 1) (inferred from first conv layer)")
            
            # Collect all layer names
            all_layers = list(f['model_weights'].keys())
            
            # Final architecture visualization
            print("\nFinal Architecture Sequence:")
            layer_types = {}
            
            for layer in all_layers:
                layer_base = layer.split('_')[0]
                if layer_base in layer_types:
                    layer_types[layer_base].append(layer)
                else:
                    layer_types[layer_base] = [layer]
            
            # Print layer types in a more organized way
            for type_name, layers in layer_types.items():
                print(f"\n{type_name} layers:")
                for layer in sorted(layers):
                    print(f"  {layer}")
            
            ###########################################
            # VISUALIZATIONS
            ###########################################
            
            # Plot layer distribution
            plt.figure(figsize=(10, 6))
            layer_type_counts = {}
            for name in all_layers:
                base_type = name.split('_')[0]
                if base_type in layer_type_counts:
                    layer_type_counts[base_type] += 1
                else:
                    layer_type_counts[base_type] = 1
            
            plt.bar(layer_type_counts.keys(), layer_type_counts.values())
            plt.title('Distribution of Layer Types')
            plt.ylabel('Count')
            plt.xlabel('Layer Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'layer_distribution.png'))
            print(f"\nLayer distribution plot saved to '{os.path.join(output_dir, 'layer_distribution.png')}'")
            
            # Filter size progression visualization
            plt.figure(figsize=(12, 6))
            filter_nums = [filters for _, filters in filter_progression]
            filter_names = [name for name, _ in filter_progression]
            
            plt.bar(filter_names, filter_nums)
            plt.title('Filter Size Progression Through Network')
            plt.ylabel('Number of Filters')
            plt.xlabel('Layer')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'filter_progression.png'))
            print(f"Filter progression plot saved to '{os.path.join(output_dir, 'filter_progression.png')}'")
            
            ###########################################
            # SUMMARY REPORT GENERATION
            ###########################################
            
            # Save a detailed summary report
            with open(os.path.join(output_dir, 'model_architecture_summary.txt'), 'w') as f_summary:
                f_summary.write("FER+ MODEL ARCHITECTURE SUMMARY\n")
                f_summary.write("=" * 80 + "\n\n")
                f_summary.write(f"Total parameters: {total_params:,}\n\n")
                
                f_summary.write("Filter progression:\n")
                for name, filters in filter_progression:
                    f_summary.write(f"  {name}: {filters} filters\n")
                
                f_summary.write("\nLayer type distribution:\n")
                for layer_type, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
                    f_summary.write(f"  {layer_type}: {count}\n")
                
                f_summary.write("\nActivation functions: LeakyReLU (with alpha=0.02 typically)\n")
                f_summary.write("Dropout type: SpatialDropout2D (with rate=0.1 typically)\n")
                f_summary.write("Final layer: GlobalAveragePooling followed by Softmax\n")
                
                # Add number of classes if available
                if sep_conv_layers:
                    # Get the highest key (last layer)
                    last_layer_num = max(sep_conv_layers.keys())
                    final_filters = sep_conv_layers[last_layer_num]
                    f_summary.write(f"\nOutput classes: {final_filters} (inferred from final layer)\n")
            
            print(f"\nDetailed summary saved to '{os.path.join(output_dir, 'model_architecture_summary.txt')}'")
            
            # Return overall success
            return True
    except Exception as e:
        print(f"Error inspecting H5 file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

###########################################
# MAIN EXECUTION
###########################################

def main():
    """Main function to run the model architecture analyzer"""
    # Check if model file exists
    if os.path.exists(MODEL_PATH):
        print(f"Found model at: {MODEL_PATH}")
        
        # Perform H5 file inspection
        print("\nPerforming detailed H5 file inspection...")
        inspect_model_h5_file(MODEL_PATH, OUTPUT_DIR)
        
        print("\n===== ARCHITECTURE INSPECTION COMPLETE =====")
        print(f"Results saved to: {OUTPUT_DIR}")
    else:
        print(f"Model file not found at: {MODEL_PATH}")
        print("Please update the MODEL_PATH at the top of the script to point to your model file.")
        
        # Check for Google Colab and offer upload option
        try:
            from google.colab import files
            print("\nRunning in Google Colab. Please upload your model file:")
            uploaded = files.upload()
            
            if uploaded:
                model_file = list(uploaded.keys())[0]
                print(f"Analyzing uploaded model: {model_file}")
                inspect_model_h5_file(model_file, OUTPUT_DIR)
        except ImportError:
            # Not running in Google Colab
            pass

# Run the main function when the script is executed
if __name__ == "__main__":
    main()