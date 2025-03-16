# bias_mitigation_approaches/multi_task_learning/multitask_fairness.py

import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import datetime

# Import local modules
from config import *
from model.multi_task_model import create_multi_task_model
from model.fine_tuning import progressive_fine_tuning
from utils.data_loader import load_combined_rafdb_dataset, prepare_multi_task_data, demographic_stratified_split
from utils.metrics import evaluate_multi_task_model
from utils.visualization import plot_training_history, visualize_evaluation_results, plot_fairness_trends

# Set up GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using {len(gpus)} GPU(s)")
else:
    print("Using CPU")

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed} for reproducibility")

def main():
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create timestamped folder for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 80)
    print("MULTI-TASK LEARNING FOR BIAS MITIGATION")
    print("=" * 80)
    print(f"Base model: {BASE_MODEL_PATH}")
    print(f"Output model: {MULTI_TASK_MODEL_OUTPUT_PATH}")
    print(f"Results directory: {run_dir}")
    print("=" * 80)
    
    # 1. Load combined dataset (original + augmented)
    print("\nLoading RAF-DB dataset (original + augmented)...")
    train_images, train_emotions, train_gender, train_age, train_demo_info = load_combined_rafdb_dataset('train', balance_demographics=True)
    test_images, test_emotions, test_gender, test_age, test_demo_info = load_combined_rafdb_dataset('test')
    
    # 2. Split training data into train and validation sets with demographic stratification
    print("\nSplitting data into train and validation sets with demographic stratification...")
    X_train, X_val, e_train, e_val, g_train, g_val, a_train, a_val = demographic_stratified_split(
        train_images, train_emotions, train_gender, train_age,
        val_split=VALIDATION_SPLIT
    )
    
    # Extract validation demographic info
    val_demo_info = [train_demo_info[i] for i in range(len(train_demo_info)) 
                     if i < len(train_emotions) and train_emotions[i] in e_val]
    
    # 3. Prepare data for multi-task learning
    print("\nPreparing data for multi-task learning...")
    train_data = prepare_multi_task_data(X_train, e_train, g_train, a_train)
    val_data = prepare_multi_task_data(X_val, e_val, g_val, a_val)
    test_data = prepare_multi_task_data(test_images, test_emotions, test_gender, test_age)
    
    # Print data shapes
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Validation data: {X_val.shape[0]} samples")
    print(f"Test data: {test_images.shape[0]} samples")
    
    # Print demographic distributions
    print("\nDemographic distribution in training set:")
    print(f"Gender distribution: {np.bincount(g_train)}")
    print(f"Age distribution: {np.bincount(a_train)}")
    
    print("\nDemographic distribution in validation set:")
    print(f"Gender distribution: {np.bincount(g_val)}")
    print(f"Age distribution: {np.bincount(a_val)}")
    
    print("\nDemographic distribution in test set:")
    print(f"Gender distribution: {np.bincount(test_gender)}")
    print(f"Age distribution: {np.bincount(test_age)}")
    
    # 4. Create multi-task model
    print("\nCreating multi-task model from pretrained FER+ model...")
    model, feature_extractor = create_multi_task_model()
    
    # 5. Train with progressive fine-tuning
    print("\nStarting progressive fine-tuning...")
    history, fairness_metrics = progressive_fine_tuning(
        model, 
        feature_extractor, 
        train_data, 
        val_data,
        val_demographic_info=val_demo_info
    )
    
    # 6. Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, os.path.join(run_dir, 'training_history'))
    
    # 7. Plot fairness trends
    print("\nPlotting fairness trends...")
    plot_fairness_trends(fairness_metrics, os.path.join(run_dir, 'fairness_trends'))
    
    # 8. Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss = model.evaluate(test_data[0], test_data[1], verbose=1)
    
    # 9. Generate comprehensive evaluation results
    print("\nGenerating comprehensive evaluation results...")
    evaluation_results = evaluate_multi_task_model(model, test_data[0], test_data[1])
    
    # Print key results
    print("\nTest Set Results:")
    print(f"Overall Emotion Recognition Accuracy: {evaluation_results['emotion_metrics']['accuracy']:.4f}")
    print(f"Gender Fairness Score: {evaluation_results['gender_fairness']['fairness_score']:.4f}")
    print(f"Age Fairness Score: {evaluation_results['age_fairness']['fairness_score']:.4f}")
    print(f"Intersectional Fairness Score: {evaluation_results['intersectional_metrics']['fairness_score']:.4f}")
    print(f"Gender Prediction Accuracy: {evaluation_results['gender_accuracy']:.4f}")
    print(f"Age Prediction Accuracy: {evaluation_results['age_accuracy']:.4f}")
    
    # 10. Visualize evaluation results
    print("\nVisualizing evaluation results...")
    visualize_evaluation_results(evaluation_results, os.path.join(run_dir, 'evaluation'))
    
    # 11. Save evaluation results to CSV
    print("\nSaving evaluation results...")
    
    # Create summary DataFrame
    summary_data = {
        'Metric': [
            'Overall Accuracy',
            'Emotion Fairness Score',
            'Gender Fairness Score',
            'Age Fairness Score',
            'Intersectional Fairness Score',
            'Gender Prediction Accuracy',
            'Age Prediction Accuracy',
            'Gender AUC',
            'Age AUC'
        ],
        'Value': [
            evaluation_results['emotion_metrics']['accuracy'],
            evaluation_results['emotion_metrics']['emotion_fairness'],
            evaluation_results['gender_fairness']['fairness_score'],
            evaluation_results['age_fairness']['fairness_score'],
            evaluation_results['intersectional_metrics']['fairness_score'],
            evaluation_results['gender_accuracy'],
            evaluation_results['age_accuracy'],
            evaluation_results.get('gender_auc', 0),
            evaluation_results.get('age_auc', 0)
        ]
    }
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(run_dir, 'evaluation_summary.csv'), index=False)
    
    # 12. Save per-emotion accuracies
    emotion_df = pd.DataFrame({
        'Emotion': list(evaluation_results['emotion_metrics']['class_accuracies'].keys()),
        'Accuracy': list(evaluation_results['emotion_metrics']['class_accuracies'].values())
    })
    emotion_df.to_csv(os.path.join(run_dir, 'emotion_accuracies.csv'), index=False)
    
    # 13. Save demographic group accuracies
    gender_df = pd.DataFrame({
        'Gender': list(evaluation_results['gender_fairness']['group_accuracies'].keys()),
        'Accuracy': list(evaluation_results['gender_fairness']['group_accuracies'].values())
    })
    gender_df.to_csv(os.path.join(run_dir, 'gender_group_accuracies.csv'), index=False)
    
    age_df = pd.DataFrame({
        'Age_Group': list(evaluation_results['age_fairness']['group_accuracies'].keys()),
        'Accuracy': list(evaluation_results['age_fairness']['group_accuracies'].values())
    })
    age_df.to_csv(os.path.join(run_dir, 'age_group_accuracies.csv'), index=False)
    
    # 14. Save intersectional accuracies
    intersectional_data = []
    for group_name, metrics in evaluation_results['intersectional_metrics']['intersectional_accuracies'].items():
        intersectional_data.append({
            'Demographic_Group': group_name,
            'Accuracy': metrics['accuracy'],
            'Count': metrics['count']
        })
    
    intersectional_df = pd.DataFrame(intersectional_data)
    intersectional_df.to_csv(os.path.join(run_dir, 'intersectional_accuracies.csv'), index=False)
    
    # 15. Save equalized odds metrics
    if 'equalized_odds' in evaluation_results['gender_fairness']:
        odds_data = []
        for emotion, metrics in evaluation_results['gender_fairness']['equalized_odds'].items():
            odds_data.append({
                'Emotion': emotion,
                'TPR_Difference': metrics['tpr_difference'],
                'FPR_Difference': metrics['fpr_difference'],
                'Equalized_Odds_Score': metrics['equalized_odds_score']
            })
        
        odds_df = pd.DataFrame(odds_data)
        odds_df.to_csv(os.path.join(run_dir, 'gender_equalized_odds.csv'), index=False)
    
    # 16. Save demographic parity metrics
    if 'demographic_parity' in evaluation_results['gender_fairness']:
        parity_data = []
        for emotion, metrics in evaluation_results['gender_fairness']['demographic_parity'].items():
            parity_data.append({
                'Emotion': emotion,
                'Difference': metrics['difference'],
                'Ratio': metrics['ratio']
            })
        
        parity_df = pd.DataFrame(parity_data)
        parity_df.to_csv(os.path.join(run_dir, 'gender_demographic_parity.csv'), index=False)
    
    # 17. Save model architecture summary
    with open(os.path.join(run_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # 18. Save training parameters
    params_data = {
        'Parameter': [
            'Initial Learning Rate Phase 1',
            'Initial Learning Rate Phase 2',
            'Batch Size',
            'Gradient Clip Norm',
            'Emotion Loss Weight',
            'Gender Loss Weight',
            'Age Loss Weight',
            'Max Epochs',
            'Phase 1 Epochs',
            'Warmup Epochs Phase 1',
            'Warmup Epochs Phase 2',
            'Regularization Rate',
            'Dropout Rate'
        ],
        'Value': [
            INITIAL_LR_PHASE1,
            INITIAL_LR_PHASE2,
            BATCH_SIZE,
            GRADIENT_CLIP_NORM,
            EMOTION_LOSS_WEIGHT,
            GENDER_LOSS_WEIGHT,
            AGE_LOSS_WEIGHT,
            MAX_EPOCHS,
            PHASE1_EPOCHS,
            WARMUP_EPOCHS_PHASE1,
            WARMUP_EPOCHS_PHASE2,
            REGULARIZATION_RATE,
            SPATIAL_DROPOUT_RATE
        ]
    }
    
    params_df = pd.DataFrame(params_data)
    params_df.to_csv(os.path.join(run_dir, 'training_parameters.csv'), index=False)
    
    print(f"\nTraining and evaluation complete!")
    print(f"Model saved to {MULTI_TASK_MODEL_OUTPUT_PATH}")
    print(f"Results saved to {run_dir}")

if __name__ == "__main__":
    main()