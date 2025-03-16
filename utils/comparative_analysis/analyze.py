# utils/comparative_analysis/analyze.py

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Fix imports to work when run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import local modules with absolute imports
from metrics import (
    calculate_overall_metrics, 
    calculate_emotion_accuracies,
    calculate_demographic_metrics,
    RAFDB_EMOTIONS,
    FERPLUS_EMOTIONS,
    RAFDB_TO_FERPLUS,
    AGE_GROUPS,
    GENDER_LABELS
)
from plots import (
    plot_emotion_accuracies,
    plot_demographic_accuracies,
    plot_emotion_by_demographic,
    plot_confusion_matrix,
    plot_data_distribution,
    plot_comparison_results,
    plot_demographic_group_accuracies
)

# Path configurations based on the observed directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # BIAS MITIGATIONS/utils
MODELS_DIR = os.path.join('resources', 'models')
DATASET_DIR = os.path.join('resources', 'dataset', 'raf-db', 'dataset')
RAFDB_DIR = os.path.join('resources', 'dataset', 'raf-db')
RESULTS_DIR = os.path.join(current_dir, 'results')

# Plot control switch - set this to False to disable plots when testing
GENERATE_PLOTS = False
SHOW_PLOTS = False

def load_test_labels():
    """Load test labels from RAF-DB dataset"""
    labels_path = os.path.join(RAFDB_DIR, 'labels', 'test_labels.csv')
    df = pd.read_csv(labels_path)
    return df

def load_and_preprocess_images(test_labels_df, base_path):
    """Load and preprocess images from RAF-DB dataset"""
    images = []
    processed_files = []

    print("Loading and preprocessing images...")
    for idx, row in test_labels_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing image {idx}/{len(test_labels_df)}")

        img_path = os.path.join(base_path, str(row['label']), row['image'])
        try:
            img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            images.append(img_array)
            processed_files.append(row['image'])
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

    return np.array(images), processed_files

def analyze_model(model_path, model_name, test_labels, X_test, generate_plots=True, show_plots=True):
    """Analyze a single model's performance"""
    # Create output directory for this model
    model_results_dir = os.path.join(RESULTS_DIR, model_name.lower().replace(" ", "_"))
    os.makedirs(model_results_dir, exist_ok=True)
    
    # Create plots directory for this model if generating plots
    plots_dir = None
    if generate_plots:
        plots_dir = os.path.join(model_results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Check model output shape to handle both 7 and 8 class models
    num_classes = model.output_shape[1]
    
    # Generate predictions
    predictions = model.predict(X_test, verbose=0)
    
    # Get FER+ predictions (0-based)
    y_pred_ferplus = np.argmax(predictions, axis=1)
    
    # Get true RAF-DB labels
    y_true = test_labels['label'].values
    
    # Convert RAF-DB labels to FER+ format (0-based) for some evaluations
    y_true_fer = np.array([RAFDB_TO_FERPLUS[label] for label in y_true])
    
    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(y_true_fer, y_pred_ferplus)
    
    # Calculate emotion-specific metrics
    emotion_accuracies = calculate_emotion_accuracies(y_true, y_pred_ferplus)
    
    # Gender analysis
    gender_metrics = calculate_demographic_metrics(
        y_true, y_pred_ferplus, test_labels['Gender'].values, GENDER_LABELS, "Gender"
    )
    
    # Age analysis
    age_metrics = calculate_demographic_metrics(
        y_true, y_pred_ferplus, test_labels['Age_Group'].values, AGE_GROUPS, "Age"
    )
    
    # Generate plots if enabled
    if generate_plots:
        # Extract just accuracy values for compatibility with existing plot functions
        emotion_acc_only = {k: v['accuracy'] for k, v in emotion_accuracies.items()}
        
        plot_emotion_accuracies(emotion_acc_only, 
                                title=f"{model_name} - Emotion Recognition Accuracy", 
                                save_dir=plots_dir,
                                show_plots=show_plots)
        
        plot_confusion_matrix(y_true_fer, y_pred_ferplus, 
                             title=f"{model_name} - Confusion Matrix", 
                             save_dir=plots_dir,
                             show_plots=show_plots)
        
        plot_demographic_accuracies(gender_metrics, 
                                   title=f"{model_name} - Gender Group Accuracies", 
                                   save_dir=plots_dir,
                                   show_plots=show_plots)
        
        plot_emotion_by_demographic(gender_metrics, 
                                   title=f"{model_name} - Emotion Accuracy by Gender", 
                                   save_dir=plots_dir,
                                   show_plots=show_plots)
        
        plot_demographic_accuracies(age_metrics, 
                                   title=f"{model_name} - Age Group Accuracies", 
                                   save_dir=plots_dir,
                                   show_plots=show_plots)
        
        plot_emotion_by_demographic(age_metrics, 
                                   title=f"{model_name} - Emotion Accuracy by Age", 
                                   save_dir=plots_dir,
                                   show_plots=show_plots)
    
    # Save results
    demographic_metrics = [gender_metrics, age_metrics]
    results = {
        "model_name": model_name,
        "overall": overall_metrics,
        "emotion_accuracies": emotion_accuracies,
        "demographic": demographic_metrics,
        "num_classes": num_classes
    }
    
    # Save results to file
    results_file = os.path.join(model_results_dir, "metrics.json")
    with open(results_file, 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    return results

def print_comparison_summary_table_rich(results):
    """
    Print a formatted summary table comparing key metrics across all models using rich library
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    # Initialize console
    console = Console()
    
    # Get model names
    models = list(results.keys())
    
    # Create the main table
    main_table = Table(title="MODEL COMPARISON SUMMARY TABLE", 
                       title_style="bold white", 
                       header_style="bold cyan",
                       border_style="white",
                       expand=True)
    
    # Add columns - with wider width for model names
    main_table.add_column("Metric (F1-Score %)", style="bold yellow")
    for model in models:
        # Don't truncate model names
        main_table.add_column(model, justify="right")
    
    # Create function to highlight best value
    def highlight_best(values, higher_is_better=True):
        """Highlight the best value in a row"""
        float_values = [float(v.replace('%', '')) for v in values]
        best_idx = float_values.index(max(float_values)) if higher_is_better else float_values.index(min(float_values))
        
        result = values.copy()
        result[best_idx] = Text(values[best_idx], style="bold green")
        return result
    
    # Overall metrics section
    main_table.add_row("OVERALL METRICS", *["" for _ in models], style="bold magenta")
    
    # Overall F1-Score
    f1_row = ["Overall F1-Score"]
    for model in models:
        f1 = results[model]["overall"]["f1_score"] * 100
        f1_row.append(f"{f1:.2f}%")
    main_table.add_row(f1_row[0], *highlight_best(f1_row[1:]))
    
    # Overall Accuracy (as a reference)
    acc_row = ["Overall Accuracy"]
    for model in models:
        acc = results[model]["overall"]["accuracy"] * 100
        acc_row.append(f"{acc:.2f}%")
    main_table.add_row(acc_row[0], *highlight_best(acc_row[1:]))
    
    # Add section separator
    main_table.add_section()
    
    # GENDER SECTION
    main_table.add_row("GENDER GROUP", *["" for _ in models], style="bold magenta")
    
    # Gender fairness
    gender_fairness = ["Gender Fairness Score"]
    for model in models:
        for demo_metric in results[model]["demographic"]:
            if demo_metric["attribute"] == "Gender":
                fairness = demo_metric["fairness_score"] * 100
                gender_fairness.append(f"{fairness:.2f}%")
                break
    
    main_table.add_row(gender_fairness[0], *highlight_best(gender_fairness[1:]))
    
    # Gender groups
    for gender_key, gender_name in GENDER_LABELS.items():
        gender_group_row = [gender_name]
        for model in models:
            for demo_metric in results[model]["demographic"]:
                if demo_metric["attribute"] == "Gender":
                    if gender_name in demo_metric["metrics"]:
                        f1 = demo_metric["metrics"][gender_name]["f1_score"] * 100
                        gender_group_row.append(f"{f1:.2f}%")
                    else:
                        gender_group_row.append("N/A")
                    break
        
        main_table.add_row(gender_group_row[0], *highlight_best(gender_group_row[1:]))
    
    # Add section separator
    main_table.add_section()
    
    # AGE SECTION
    main_table.add_row("AGE GROUP", *["" for _ in models], style="bold magenta")
    
    # Age fairness
    age_fairness = ["Age Fairness Score"]
    for model in models:
        for demo_metric in results[model]["demographic"]:
            if demo_metric["attribute"] == "Age":
                fairness = demo_metric["fairness_score"] * 100
                age_fairness.append(f"{fairness:.2f}%")
                break
    
    main_table.add_row(age_fairness[0], *highlight_best(age_fairness[1:]))
    
    # Age groups
    for age_key, age_name in AGE_GROUPS.items():
        age_group_row = [age_name]
        for model in models:
            for demo_metric in results[model]["demographic"]:
                if demo_metric["attribute"] == "Age":
                    if age_name in demo_metric["metrics"]:
                        f1 = demo_metric["metrics"][age_name]["f1_score"] * 100
                        age_group_row.append(f"{f1:.2f}%")
                    else:
                        age_group_row.append("N/A")
                    break
        
        main_table.add_row(age_group_row[0], *highlight_best(age_group_row[1:]))
    
    # Add section separator
    main_table.add_section()
    
    # EMOTION SECTION
    main_table.add_row("EMOTION RECOGNITION", *["" for _ in models], style="bold magenta")
    
    # Get all unique emotions across all models
    all_emotions = set()
    for model in models:
        all_emotions.update(results[model]["emotion_accuracies"].keys())
    
    # Sort emotions in a standard order if possible
    standard_order = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
    emotions = [e for e in standard_order if e in all_emotions]
    # Add any remaining emotions
    emotions.extend([e for e in all_emotions if e not in emotions])
    
    # Print each emotion row
    for emotion in emotions:
        emotion_row = [emotion]
        for model in models:
            if emotion in results[model]["emotion_accuracies"]:
                f1 = results[model]["emotion_accuracies"][emotion]["f1_score"] * 100
                emotion_row.append(f"{f1:.2f}%")
            else:
                emotion_row.append("N/A")
        
        main_table.add_row(emotion_row[0], *highlight_best(emotion_row[1:]))
    
    # Print the table
    console.print("\n")
    console.print(main_table)
    console.print("\n")

def print_comparison_summary_table(results):
    """
    Print a formatted summary table comparing key metrics across all models
    """
    models = list(results.keys())
    # Make the table wider to accommodate longer model names
    table_width = 120  # Increased from 100
    
    print(f"\n{'='*table_width}")
    print(f"{'MODEL COMPARISON SUMMARY TABLE':^{table_width}}")
    print(f"{'='*table_width}")
    
    # Header row - use more space for each model column
    header = f"{'Metric (F1-Score %)':25}"
    for model in models:
        # Don't truncate model names, give them 30 chars instead of 25
        header += f"{model:^30}"
    print(header)
    print('-' * table_width)
    
    # OVERALL METRICS SECTION
    print(f"\n{' OVERALL METRICS ':=^{table_width}}")
    
    # Overall F1-Score
    f1_row = f"{'Overall F1-Score':<25}"
    for model in models:
        f1 = results[model]["overall"]["f1_score"] * 100
        f1_row += f"{f1:27.2f}% "  # Adjusted spacing
    print(f1_row)
    
    # Overall Accuracy (as reference)
    acc_row = f"{'Overall Accuracy':<25}"
    for model in models:
        acc = results[model]["overall"]["accuracy"] * 100
        acc_row += f"{acc:27.2f}% "  # Adjusted spacing
    print(acc_row)
    
    # GENDER SECTION
    print(f"\n{' GENDER GROUP ':=^{table_width}}")
    
    # Gender fairness
    gender_row = f"{'Gender Fairness Score':<25}"
    for model in models:
        for demo_metric in results[model]["demographic"]:
            if demo_metric["attribute"] == "Gender":
                fairness = demo_metric["fairness_score"] * 100
                gender_row += f"{fairness:27.2f}% "  # Adjusted spacing
                break
    print(gender_row)
    
    # Gender groups
    for gender_key, gender_name in GENDER_LABELS.items():
        gender_group_row = f"{gender_name:<25}"
        for model in models:
            for demo_metric in results[model]["demographic"]:
                if demo_metric["attribute"] == "Gender":
                    if gender_name in demo_metric["metrics"]:
                        f1 = demo_metric["metrics"][gender_name]["f1_score"] * 100
                        gender_group_row += f"{f1:27.2f}% "  # Adjusted spacing
                    else:
                        gender_group_row += f"{'N/A':^30}"  # Adjusted spacing
                    break
        print(gender_group_row)
    
    # AGE SECTION
    print(f"\n{' AGE GROUP ':=^{table_width}}")
    
    # Age fairness
    age_row = f"{'Age Fairness Score':<25}"
    for model in models:
        for demo_metric in results[model]["demographic"]:
            if demo_metric["attribute"] == "Age":
                fairness = demo_metric["fairness_score"] * 100
                age_row += f"{fairness:27.2f}% "  # Adjusted spacing
                break
    print(age_row)
    
    # Age groups
    for age_key, age_name in AGE_GROUPS.items():
        age_group_row = f"{age_name:<25}"
        for model in models:
            for demo_metric in results[model]["demographic"]:
                if demo_metric["attribute"] == "Age":
                    if age_name in demo_metric["metrics"]:
                        f1 = demo_metric["metrics"][age_name]["f1_score"] * 100
                        age_group_row += f"{f1:27.2f}% "  # Adjusted spacing
                    else:
                        age_group_row += f"{'N/A':^30}"  # Adjusted spacing
                    break
        print(age_group_row)
    
    # EMOTION SECTION
    print(f"\n{' EMOTION RECOGNITION ':=^{table_width}}")
    
    # Get all unique emotions across all models
    all_emotions = set()
    for model in models:
        all_emotions.update(results[model]["emotion_accuracies"].keys())
    
    # Sort emotions in a standard order if possible
    standard_order = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
    emotions = [e for e in standard_order if e in all_emotions]
    # Add any remaining emotions
    emotions.extend([e for e in all_emotions if e not in emotions])
    
    # Print each emotion row
    for emotion in emotions:
        emotion_row = f"{emotion:<25}"
        for model in models:
            if emotion in results[model]["emotion_accuracies"]:
                f1 = results[model]["emotion_accuracies"][emotion]["f1_score"] * 100
                emotion_row += f"{f1:27.2f}% "  # Adjusted spacing
            else:
                emotion_row += f"{'N/A':^30}"  # Adjusted spacing
        print(emotion_row)
    
    print('=' * table_width)
    print(f"{'END OF COMPARISON':^{table_width}}")
    print('=' * table_width)

def compare_models(models_config, test_labels, X_test, generate_plots=True, show_plots=True, use_rich_table=True):
    """Compare multiple models on bias and performance metrics"""
    results = {}
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create a plots directory for comparison plots if generating plots
    comparison_plots_dir = None
    if generate_plots:
        comparison_plots_dir = os.path.join(RESULTS_DIR, 'comparison_plots')
        os.makedirs(comparison_plots_dir, exist_ok=True)
    
    # Analyze each model (silently)
    print("Analyzing models...")
    for model_name, model_path in models_config.items():
        print(f"  - Loading {model_name}...")
        model = tf.keras.models.load_model(model_path)
        
        print(f"  - Generating predictions...")
        predictions = model.predict(X_test, verbose=0)
        
        # Get FER+ predictions (0-based)
        y_pred_ferplus = np.argmax(predictions, axis=1)
        
        # Get true RAF-DB labels
        y_true = test_labels['label'].values
        
        # Convert RAF-DB labels to FER+ format (0-based) for some evaluations
        y_true_fer = np.array([RAFDB_TO_FERPLUS[label] for label in y_true])
        
        # Calculate metrics
        overall_metrics = calculate_overall_metrics(y_true_fer, y_pred_ferplus)
        emotion_accuracies = calculate_emotion_accuracies(y_true, y_pred_ferplus)
        
        gender_metrics = calculate_demographic_metrics(
            y_true, y_pred_ferplus, test_labels['Gender'].values, GENDER_LABELS, "Gender"
        )
        
        age_metrics = calculate_demographic_metrics(
            y_true, y_pred_ferplus, test_labels['Age_Group'].values, AGE_GROUPS, "Age"
        )
        
        # Store results
        results[model_name] = {
            "model_name": model_name,
            "overall": overall_metrics,
            "emotion_accuracies": emotion_accuracies,
            "demographic": [gender_metrics, age_metrics],
            "num_classes": model.output_shape[1]
        }
    
    # Visualize the results - separate generate_plots and show_plots logic
    if generate_plots or show_plots:
        print("\nGenerating visualizations...")
        
        # Determine save directory based on generate_plots flag
        save_dir = comparison_plots_dir if generate_plots else None
        
        # Overall accuracy comparison
        plot_comparison_results(results, "Overall F1-Score", 
                                save_dir=save_dir,
                                show_plots=show_plots)
        
        # Fairness comparisons
        plot_comparison_results(results, "Fairness-Gender", 
                                save_dir=save_dir,
                                show_plots=show_plots)
        plot_comparison_results(results, "Fairness-Age", 
                                save_dir=save_dir,
                                show_plots=show_plots)
        
        # Detailed demographic group accuracies
        plot_demographic_group_accuracies(results, "Gender", 
                                         save_dir=save_dir,
                                         show_plots=show_plots)
        plot_demographic_group_accuracies(results, "Age", 
                                         save_dir=save_dir,
                                         show_plots=show_plots)
    
    # Print comparative summary table
    if use_rich_table:
        try:
            print_comparison_summary_table_rich(results)
        except ImportError:
            print("\nNote: 'rich' library not found. Using standard table format.")
            print("To install rich: pip install rich")
            print_comparison_summary_table(results)
    else:
        print_comparison_summary_table(results)
    
    return results

def main():
    """Main function to run the comparative analysis"""
    print("Starting comparative analysis...")
    
    # Check if rich is installed
    try:
        import rich
        use_rich_table = True
        print("Rich library detected - will use enhanced table formatting")
    except ImportError:
        use_rich_table = False
        print("Note: For better table formatting, install rich library: pip install rich")
    
    # 1. Load test labels
    test_labels = load_test_labels()
    print(f"Loaded {len(test_labels)} test samples")
    
    # 2. Display dataset distribution if plots are enabled
    if GENERATE_PLOTS or SHOW_PLOTS:
        # Create dataset directory for dataset plots
        dataset_plots_dir = os.path.join(RESULTS_DIR, 'dataset_plots')
        os.makedirs(dataset_plots_dir, exist_ok=True) if GENERATE_PLOTS else None
        plot_data_distribution(test_labels, 
                              save_dir=dataset_plots_dir if GENERATE_PLOTS else None,
                              show_plots=SHOW_PLOTS)
    
    # 3. Load and preprocess images
    test_base_path = os.path.join(DATASET_DIR, 'test')
    X_test, processed_files = load_and_preprocess_images(test_labels, test_base_path)
    print(f"Loaded {len(X_test)} test images")
    
    # 4. Configure models to analyzeclear
    models_config = {
        # Add models to include on the comparative analysis
        # "Baseline": os.path.join(MODELS_DIR, 'baseline-ferplus-model.h5'),
        # "Emotion-Based Augmentation (without layer freezing)": os.path.join(MODELS_DIR, 'emotion_augmentation_finetuned_model (without layer freezing).h5'),
        # "Demographic-Based Augmentation (without layer freezing)": os.path.join(MODELS_DIR, 'demographic_augmentation_finetuned_model (without layer freezing).h5'),
        # "Emotion-Based Augmentation": os.path.join(MODELS_DIR, 'emotion_augmentation_finetuned_model.h5'),
        # "Demographic-Based Augmentation": os.path.join(MODELS_DIR, 'demographic_augmentation_finetuned_model.h5'),
        

        "Multi-Task Learning (Fairness-aware)": os.path.join(MODELS_DIR, 'multi_task_model.h5'),



        # You can add more models here
    }
    
    # 5. Compare models
    results = compare_models(
        models_config, 
        test_labels, 
        X_test,
        generate_plots=GENERATE_PLOTS,
        show_plots=SHOW_PLOTS,
        use_rich_table=use_rich_table
    )
    
    print("Comparative analysis complete!")
    return results

if __name__ == "__main__":
    main()