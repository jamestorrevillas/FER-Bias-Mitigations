# Multi-Task Learning for Facial Emotion Recognition Bias Mitigation

This implementation adds demographic-aware multi-task learning to facial emotion recognition to mitigate biases across demographic groups.

## Approach

The multi-task learning approach (also called "Attribute-Aware" or "Fairness Through Awareness") explicitly incorporates demographic attributes as auxiliary tasks during training. By jointly learning emotion recognition along with gender and age classification, the model develops internal representations that are aware of demographic characteristics.

### Key Features

- Uses pretrained FER+ model as the base feature extractor
- Adds parallel task heads for gender and age prediction
- Employs progressive fine-tuning strategy
- Balances loss weights between primary and auxiliary tasks
- Combines original and augmented data for better balance

## Directory Structure

```
multi_task_learning/
├── model/
│   ├── multi_task_model.py          # Multi-task model architecture
│   └── fine_tuning.py               # Progressive fine-tuning logic
├── utils/
│   ├── data_loader.py               # Dataset loading and preparation
│   ├── metrics.py                   # Fairness and performance metrics
│   └── visualization.py             # Result visualization utilities
├── config.py                        # Configuration parameters
├── train.py                         # Main training script
└── README.md                        # This documentation
```

## Usage

To train the multi-task model:

```bash
cd bias_mitigation_approaches
python -m multi_task_learning.train
```

## Fine-tuning Approach

The training process follows a two-phase progressive fine-tuning approach:

1. **Phase 1: Train Task-Specific Heads**
   - Freeze the pretrained feature extractor layers
   - Train only the new task-specific heads
   - Use higher learning rate for faster convergence

2. **Phase 2: Fine-tune Full Network**
   - Unfreeze all layers
   - Use much lower learning rate to prevent catastrophic forgetting
   - Apply early stopping based on validation accuracy

## Data

The implementation uses a combination of:

- Original RAF-DB dataset (with demographic annotations)
- Augmented RAF-DB dataset (for better demographic balance)

## Evaluation Metrics

The model is evaluated on:

- Overall emotion recognition accuracy
- Per-emotion class accuracies
- Emotion fairness score (min/max ratio of emotion accuracies)
- Gender fairness score (min/max ratio of male vs. female accuracy)
- Age fairness score (min/max ratio across age groups)
- Auxiliary task accuracies (gender and age prediction)

## Results

Evaluation results and visualizations are saved to the `results` directory and include:

- Training history plots
- Confusion matrix
- Group accuracy charts
- Fairness metrics
- Emotion-by-demographic heatmaps

```

These files provide a complete implementation of the multi-task learning approach for bias mitigation in facial emotion recognition. Let me know if you need any adjustments or have questions about specific parts of the implementation!
