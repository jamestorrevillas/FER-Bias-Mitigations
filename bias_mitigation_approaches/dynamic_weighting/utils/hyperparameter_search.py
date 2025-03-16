# bias_mitigation_approaches/dynamic_weighting/utils/hyperparameter_search.py

"""
Hyperparameter search script for dynamic cross-dataset weighting approach.
This script runs multiple fine-tuning sessions with different hyperparameter
configurations to find the optimal settings for bias mitigation.

Usage:
    python hyperparameter_search.py

Output:
    - Search results saved to results/hyperparameter_search directory
    - Best hyperparameter configuration identified and saved
"""

import os
import sys
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time
from datetime import datetime

# Import local modules
from utils import config
from dynamic_fine_tuning import main as run_fine_tuning

# Create results directory
SEARCH_RESULTS_DIR = os.path.join(config.RESULTS_DIR, 'hyperparameter_search')
os.makedirs(SEARCH_RESULTS_DIR, exist_ok=True)

# Define hyperparameter search space
def run_hyperparameter_search(full_grid=False):
    """
    Run hyperparameter search with configurable grid density
    
    Args:
        full_grid: Whether to use the full grid or a reduced grid
    """
    if full_grid:
        # Full grid for comprehensive search
        param_grid = {
            'INITIAL_WEIGHT_MULTIPLIER': [1.5, 2.0, 2.5, 3.0],
            'WEIGHT_DECAY_FACTOR': [0.9, 0.92, 0.95, 0.98],
            'FAIRNESS_IMPROVEMENT_THRESHOLD': [0.002, 0.005, 0.01, 0.02],
            'INTERSECTION_WEIGHT_MULTIPLIER': [1.5, 2.0, 2.5, 3.0],
            'MAX_INTERSECTION_MULTIPLIER': [3.0, 4.0, 5.0]
        }
    else:
        # Reduced grid for faster exploration
        param_grid = {
            'INITIAL_WEIGHT_MULTIPLIER': [1.5, 2.5],
            'WEIGHT_DECAY_FACTOR': [0.9, 0.95],
            'FAIRNESS_IMPROVEMENT_THRESHOLD': [0.005, 0.015],
            'INTERSECTION_WEIGHT_MULTIPLIER': [1.5, 2.5]
        }
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))
    
    print(f"Starting hyperparameter search with {len(param_combinations)} configurations")
    print(f"Results will be saved to: {SEARCH_RESULTS_DIR}")
    
    # Create timestamp for this search run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    search_log_path = os.path.join(SEARCH_RESULTS_DIR, f'search_log_{timestamp}.txt')
    results_path = os.path.join(SEARCH_RESULTS_DIR, f'search_results_{timestamp}.json')
    
    # Initialize results list
    results = []
    
    # Create initial log
    with open(search_log_path, 'w') as f:
        f.write(f"Hyperparameter search started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of configurations: {len(param_combinations)}\n")
        f.write("Parameter grid:\n")
        for param, values in param_grid.items():
            f.write(f"  {param}: {values}\n")
        f.write("\n" + "=" * 80 + "\n\n")
    
    # Run hyperparameter search
    for i, params in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"Configuration {i+1}/{len(param_combinations)}")
        print(f"{'='*80}")
        
        # Record configuration
        config_dict = dict(zip(param_names, params))
        
        # Log configuration
        with open(search_log_path, 'a') as f:
            f.write(f"Configuration {i+1}/{len(param_combinations)}\n")
            for name, value in config_dict.items():
                f.write(f"  {name} = {value}\n")
            f.write("\n")
        
        # Set configuration parameters
        for name, value in config_dict.items():
            setattr(config, name, value)
            print(f"Setting {name} = {value}")
        
        # Update output path to avoid overwriting previous runs
        config.DYNAMIC_MODEL_PATH = os.path.join(
            os.path.dirname(config.DYNAMIC_MODEL_PATH),
            f"dynamic-weighting-config_{i+1}.h5"
        )
        
        # Modify results directory
        config_results_dir = os.path.join(SEARCH_RESULTS_DIR, f'config_{i+1}')
        config.RESULTS_DIR = config_results_dir
        os.makedirs(config_results_dir, exist_ok=True)
        
        # Update checkpoint directory
        config.CHECKPOINT_DIR = os.path.join(config_results_dir, 'checkpoints')
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        
        # Run fine-tuning with current configuration
        print("\nRunning fine-tuning with current configuration...")
        start_time = time.time()
        
        try:
            result_metrics = run_fine_tuning()
            runtime = time.time() - start_time
            
            # Store results
            result_entry = {
                'config_id': i+1,
                'params': config_dict,
                'metrics': result_metrics,
                'runtime_seconds': runtime
            }
            results.append(result_entry)
            
            # Log results
            with open(search_log_path, 'a') as f:
                f.write(f"Results for configuration {i+1}:\n")
                f.write(f"  Runtime: {runtime:.2f} seconds\n")
                for metric, value in result_metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n" + "-" * 40 + "\n\n")
                
            # Save intermediate results
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
                
            print(f"Configuration {i+1} completed in {runtime:.2f} seconds")
            print(f"Results: {result_metrics}")
            
        except Exception as e:
            print(f"Error during fine-tuning: {str(e)}")
            with open(search_log_path, 'a') as f:
                f.write(f"Error in configuration {i+1}: {str(e)}\n\n")
    
    # Create final results
    print("\nHyperparameter search complete!")
    
    # Find best configuration
    if results:
        # Sort by combined fairness (primary metric)
        results_df = pd.DataFrame([
            {**{'config_id': r['config_id']}, 
             **r['params'], 
             **{f"metric_{k}": v for k, v in r['metrics'].items()},
             'runtime': r['runtime_seconds']}
            for r in results
        ])
        
        # Save results dataframe
        results_csv_path = os.path.join(SEARCH_RESULTS_DIR, f'search_results_{timestamp}.csv')
        results_df.to_csv(results_csv_path, index=False)
        
        # Find best by combined fairness
        best_config = results_df.loc[results_df['metric_combined_fairness'].idxmax()]
        
        # Log best configuration
        with open(search_log_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("BEST CONFIGURATION:\n")
            for param in param_names:
                f.write(f"  {param}: {best_config[param]}\n")
            f.write("\nBest metrics:\n")
            for col in results_df.columns:
                if col.startswith('metric_'):
                    metric_name = col[7:]  # Remove 'metric_' prefix
                    f.write(f"  {metric_name}: {best_config[col]:.4f}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        # Print best configuration
        print("\n" + "=" * 80)
        print("BEST CONFIGURATION:")
        for param in param_names:
            print(f"  {param}: {best_config[param]}")
        print("\nBest metrics:")
        for col in results_df.columns:
            if col.startswith('metric_'):
                metric_name = col[7:]  # Remove 'metric_' prefix
                print(f"  {metric_name}: {best_config[col]:.4f}")
        print("=" * 80)
        
        # Create visualization of results
        visualize_search_results(results_df, os.path.join(SEARCH_RESULTS_DIR, f'search_visualization_{timestamp}.png'))
        
        return best_config, results_df
    else:
        print("No valid results found.")
        return None, None

def visualize_search_results(results_df, save_path):
    """
    Visualize hyperparameter search results
    
    Args:
        results_df: DataFrame with search results
        save_path: Path to save visualization
    """
    # Create heatmap for pairs of parameters and their effect on combined fairness
    param_columns = [col for col in results_df.columns 
                    if col not in ['config_id', 'runtime'] 
                    and not col.startswith('metric_')]
    
    if len(param_columns) >= 2:
        # Create figure with subplots for each parameter pair
        n_pairs = len(param_columns) * (len(param_columns) - 1) // 2
        fig_cols = min(3, n_pairs)
        fig_rows = (n_pairs + fig_cols - 1) // fig_cols
        
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols*5, fig_rows*4))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Create heatmaps for parameter pairs
        pair_idx = 0
        for i, param1 in enumerate(param_columns):
            for param2 in param_columns[i+1:]:
                if pair_idx < len(axes):
                    # Create pivot table
                    pivot = results_df.pivot_table(
                        values='metric_combined_fairness', 
                        index=param1,
                        columns=param2,
                        aggfunc='mean'
                    )
                    
                    # Create heatmap
                    ax = axes[pair_idx]
                    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
                    
                    # Set labels
                    ax.set_xticks(np.arange(len(pivot.columns)))
                    ax.set_yticks(np.arange(len(pivot.index)))
                    ax.set_xticklabels(pivot.columns)
                    ax.set_yticklabels(pivot.index)
                    
                    # Rotate x labels
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax)
                    
                    # Add labels
                    ax.set_xlabel(param2)
                    ax.set_ylabel(param1)
                    ax.set_title(f'Combined Fairness: {param1} vs {param2}')
                    
                    # Add values to cells
                    for i in range(len(pivot.index)):
                        for j in range(len(pivot.columns)):
                            ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                                   ha="center", va="center", color="white")
                    
                    pair_idx += 1
        
        # Hide unused subplots
        for i in range(pair_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    # Create bar chart of fairness metrics by configuration
    plt.figure(figsize=(12, 6))
    metrics = ['metric_gender_fairness', 'metric_age_fairness', 'metric_emotion_fairness', 
              'metric_combined_fairness', 'metric_overall_accuracy']
    
    x = results_df['config_id']
    bar_width = 0.15
    positions = np.arange(len(x))
    
    for i, metric in enumerate(metrics):
        plt.bar(positions + i*bar_width - bar_width*2, 
               results_df[metric], 
               width=bar_width,
               label=metric[7:])  # Remove 'metric_' prefix
    
    plt.xlabel('Configuration ID')
    plt.ylabel('Metric Value')
    plt.title('Fairness Metrics by Configuration')
    plt.xticks(positions)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    metrics_path = os.path.join(os.path.dirname(save_path), 'fairness_by_config.png')
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    
    # Create parameter importance plot
    plt.figure(figsize=(10, 6))
    
    # Calculate correlation with combined fairness
    correlations = {}
    for param in param_columns:
        correlations[param] = results_df[param].corr(results_df['metric_combined_fairness'])
    
    # Sort by absolute correlation
    sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    params = [p[0] for p in sorted_params]
    corr_values = [p[1] for p in sorted_params]
    
    # Create bar chart
    plt.bar(params, corr_values)
    plt.xlabel('Parameter')
    plt.ylabel('Correlation with Combined Fairness')
    plt.title('Parameter Importance')
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    importance_path = os.path.join(os.path.dirname(save_path), 'parameter_importance.png')
    plt.savefig(importance_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    # Check if we should run full grid
    full_grid = len(sys.argv) > 1 and sys.argv[1].lower() == '--full'
    
    print(f"Running {'full' if full_grid else 'reduced'} hyperparameter grid search")
    best_config, results_df = run_hyperparameter_search(full_grid)
    
    # Save best config as default for future runs
    if best_config is not None:
        # Create best config file
        best_config_path = os.path.join(SEARCH_RESULTS_DIR, 'best_config.json')
        with open(best_config_path, 'w') as f:
            # Extract just the parameter values (not metrics)
            best_params = {k: v for k, v in best_config.items() 
                          if not k.startswith('metric_') and k not in ['config_id', 'runtime']}
            json.dump(best_params, f, indent=4)
        
        print(f"Best configuration saved to {best_config_path}")