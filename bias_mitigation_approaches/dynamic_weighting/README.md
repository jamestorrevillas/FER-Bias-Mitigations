# Dynamic Cross-Dataset Weighting

This implementation applies a dynamic cross-dataset weighting approach for mitigating bias in facial emotion recognition. The approach creates a feedback loop between the FER+ dataset (for training) and the RAF-DB dataset (for demographic fairness evaluation).

## Key Components

1. **Cross-Dataset Feedback Loop**: Uses RAF-DB's demographic labels to guide FER+ training
2. **Dynamic Weight Adjustment**: Automatically identifies and addresses fairness gaps
3. **Targeted Sample Weighting**: Increases importance of samples that improve fairness

## Implementation Details

The approach follows these steps:

1. **Initial Setup**:
   - Load pre-trained `baseline-ferplus-model.h5`
   - Load both FER+ and RAF-DB datasets
   - Initialize uniform sample weights

2. **Iterative Fine-tuning Cycle**:
   - **Fine-tune** for a batch on FER+ with current sample weights
   - **Evaluate** on RAF-DB to calculate demographic performance gaps
   - **Update weights** for FER+ samples based on performance gaps
   - **Repeat** until convergence or max iterations

3. **Weight Adjustment Strategy**:
   - Identify underperforming demographic-emotion combinations
   - Increase weights for corresponding samples in FER+
   - Gradually adjust weights to prevent instability

## Project Structure

```
bias_mitigation_approaches/
└── dynamic_weighting/
    ├── utils/
    │   ├── __init__.py
    │   ├── config.py              # Configuration settings
    │   ├── data_loader.py         # Dataset loading utilities
    │   ├── metrics.py             # Evaluation metrics
    │   ├── visualization.py       # Result visualization
    │   └── weight_scheduler.py    # Dynamic weight updating logic
    ├── __init__.py
    ├── dynamic_fine_tuning.py     # Main fine-tuning script 
    └── README.md                  # Documentation
```

## Usage

To run the dynamic weighting fine-tuning:

```
python dynamic_fine_tuning.py
```

This will:

1. Load the pre-trained model
2. Set up the training and evaluation datasets
3. Run the dynamic fine-tuning process
4. Save the fine-tuned model and visualization results

## Results

The approach is evaluated using:

1. **Overall Accuracy**: Performance on the primary task
2. **Fairness Metrics**:
   - Gender fairness (min/max accuracy ratio across genders)
   - Age fairness (min/max accuracy ratio across age groups)
   - Emotion fairness (min/max accuracy ratio across emotions)
3. **Intersectional Performance**: Recognition accuracy across demographic-emotion combinations

Results and visualizations are saved
