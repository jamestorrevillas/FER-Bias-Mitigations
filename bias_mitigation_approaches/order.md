# Approach Fine-Tunning Order

1. Data Augmentation
2. Static Sample Weighting
3. Dynamic Cross-Dataset Weighting
4. Multi-Task Learning with Demographic Attributes
5. Adversarial Feature Disentanglement

## Comprehensive Facial Emotion Recognition Bias Mitigation Research Plan

## Research Objective

Compare the effectiveness of different bias mitigation techniques for facial emotion recognition, evaluating both isolated technique effectiveness and cumulative improvements.

## Experimental Design

For a comprehensive evaluation, we will conduct two parallel comparison tracks:

1. **Isolated Comparison**: All approaches start from the baseline FER+ model
2. **Cumulative Comparison**: Approaches 2-5 start from the Approach #1 model (if it shows improvement)

## Approach #1: Dual-Domain Data Augmentation

## General Technique Type

**Data-Level Intervention** with **Dual-Domain Focus**  
**Paper Terminology**: "Multi-dimensional Balanced Data Augmentation"

## Core Mechanism

- Implements targeted augmentation in two complementary domains:
  1. **Emotion-domain balancing** using FER/FER+ dataset
  2. **Demographic-domain balancing** using RAF-DB dataset
- Preserves all original data while strategically generating new examples
- Addresses both emotion class imbalance and demographic representation gaps

## Implementation Details

### Phase 1: Emotion-Domain Augmentation (FER/FER+)

- **Dataset Preparation**:
  - Analyze FER+ class distribution to identify imbalance ratios
  - Apply strategic augmentation factors for each emotion class:
    - Well-represented classes (neutral, happiness): 1.0× (no augmentation)
    - Moderately underrepresented (surprise, sadness, anger): 1.4-2.5×
    - Severely underrepresented (fear, disgust, contempt): 7.5-18.0×
  - Pre-generate augmented samples using the `ferplus_generate_augmented_images.py` pipeline
  - Target reducing maximum:minimum class ratio from ~62:1 to ~3.4:1

- **Augmentation Techniques**:
  - Apply varied geometric transformations (rotation, shift, zoom, flip)
  - Use brightness and contrast adjustments for lighting robustness
  - Apply strategic class-specific probability drawing for multi-annotator agreement

### Phase 2: Demographic-Domain Augmentation (RAF-DB)

- **Dataset Preparation**:
  - Analyze RAF-DB demographic distribution across age and gender dimensions
  - Apply targeted augmentation factors for underrepresented groups:
    - Senior adults (70+): 8× augmentation factor
    - Children (0-12): 6× augmentation factor
    - Teenagers (13-19): 1× augmentation factor
    - Apply additional factors for gender imbalances where detected
  - Pre-generate augmented samples using the `raf-db_generate_augmented_images.py` pipeline

- **Intersectional Considerations**:
  - Identify critical intersections of emotion+demographic combinations
  - Apply boosted augmentation for severely underrepresented combinations (e.g., "fear+senior")
  - Track and balance distribution across combined categories

### Phase 3: Model Training with Augmented Data

- **Training Strategy**:
  1. **Base Emotion Training**: Initialize with FER+ baseline model
  2. **Emotion-Balanced Fine-tuning**: Fine-tune on emotion-augmented FER+ dataset
  3. **Demographic-Aware Refinement**: Further fine-tune using RAF-DB with demographic augmentation
  4. **Combined Validation**: Validate on both original datasets to ensure generalization

- **Implementation Process**:
  - Generate and store augmented samples for both datasets
  - Set up multi-stage training pipeline with appropriate hyperparameters
  - Monitor both emotion-based and demographic-based metrics during training

## Datasets & Models

- **Datasets**:
  - **Emotion Domain**: FER/FER+ dataset (original + augmented)
  - **Demographic Domain**: RAF-DB dataset (original + augmented)
  
- **Starting Model**:
  - Baseline FER+ model (pre-trained on original FER+ dataset)

## Evaluation Framework

- **Performance Metrics**:
  - Overall accuracy on both test sets
  - Per-emotion recognition accuracy
  - Emotion confusion matrix

- **Fairness Metrics**:
  - **Emotion Fairness Score**: Ratio of minimum:maximum emotion recognition accuracy
  - **Demographic Fairness Scores**:
    - Gender fairness (male vs. female accuracy ratio)
    - Age fairness (ratio of accuracies across age groups)
  - **Intersectional Fairness Score**: Performance across emotion-demographic combinations

- **Comparative Analysis**:
  - Pre-augmentation vs. post-augmentation performance
  - Emotion-only augmentation vs. dual-domain augmentation
  - Per-group improvement analysis

## Approach 2: Static Sample Weighting

**General Technique Type**: Training Process Intervention  
**Paper Terminology**: "Sample Weighting"  

**Core Mechanism**:

- Applies fixed sample weights during training based on observed accuracy gaps
- Gives higher importance to underperforming emotion classes

**Implementation Details**:

- Weights inversely proportional to accuracy from comparative analysis
- Applied during model.fit() through sample_weight parameter
- No architectural changes required

**Datasets & Models**:

- **Dataset**: FER+ (original + augmented)
- **Starting Models**:
  - Isolated track: Baseline FER+ model
  - Cumulative track: Approach #1 model
- **Comparison Metrics**: Overall accuracy, per-emotion accuracy, emotion fairness score

## Approach 3: Dynamic Cross-Dataset Weighting

**General Technique Type**: Training Process Intervention  
**Paper Terminology**: "Dynamic Task Weighting"  

**Core Mechanism**:

- Creates feedback loop between datasets
- Dynamically adjusts sample weights based on demographic performance

**Implementation Details**:

- Periodically evaluates demographic fairness on RAF-DB
- Updates weights in FER+ training based on evaluation
- Continuous adaptation throughout training

**Datasets & Models**:

- **Training Dataset**: FER+ (original + augmented)
- **Evaluation Dataset**: RAF-DB (for demographic feedback)
- **Starting Models**:
  - Isolated track: Baseline FER+ model
  - Cumulative track: Approach #1 model
- **Comparison Metrics**: Overall accuracy, demographic fairness scores, per-group accuracy

## Approach 4: Multi-Task Learning with Demographic Attributes

**General Technique Type**: Model Architecture Intervention  
**Paper Terminology**: "Attribute-Aware Approach" / "Fairness Through Awareness"  

**Core Mechanism**:

- Explicitly incorporates demographic attributes as additional tasks
- Creates shared representations that are aware of demographics

**Implementation Details**:

- Adds demographic prediction heads to the model
- Joint training on emotion and demographic tasks
- Uses weighted multi-task loss function

**Datasets & Models**:

- **Dataset**: RAF-DB (contains demographic labels)
- **Starting Models**:
  - Isolated track: Baseline FER+ model (with architecture modification)
  - Cumulative track: Approach #1 model (with architecture modification)
- **Comparison Metrics**: Overall accuracy, demographic fairness scores, per-group accuracy

## Approach 5: Adversarial Feature Disentanglement

**General Technique Type**: Representation-Level Intervention  
**Paper Terminology**: "Disentangled Approach"  

**Core Mechanism**:

- Actively removes demographic information from representations
- Uses adversarial training with gradient reversal

**Implementation Details**:

- Includes adversarial branches with gradient reversal layers
- Maximizes emotion accuracy while minimizing demographic predictability
- Uses confusion loss to force demographic neutrality

**Datasets & Models**:

- **Dataset**: RAF-DB (contains demographic labels)
- **Starting Models**:
  - Isolated track: Baseline FER+ model (with architecture modification)
  - Cumulative track: Approach #1 model (with architecture modification)
- **Comparison Metrics**: Overall accuracy, demographic fairness scores, demographic information leakage

## Comprehensive Evaluation Framework

Each approach will be assessed using:

1. **General Performance Metrics**:
   - Overall accuracy
   - Per-emotion accuracy
   - Confusion matrix

2. **Fairness Metrics**:
   - Emotion fairness score (min/max accuracy ratio across emotions)
   - Gender fairness score (min/max accuracy ratio across genders)
   - Age fairness score (min/max accuracy ratio across age groups)
   - Intersectional fairness (performance across gender-age combinations)

3. **Computational Efficiency**:
   - Training time
   - Model size
   - Inference speed

4. **Implementation Complexity**:
   - Lines of code required
   - Number of hyperparameters to tune
   - Training stability

## Research Visualization

The final research will include:

1. **Comparison Tables**: Direct numerical comparison of all metrics
2. **Bar Charts**: Visual comparison of key metrics across approaches
3. **Heat Maps**: Detailed visualization of per-group performance
4. **Ablation Studies**: Analysis of contribution of each component
5. **Statistical Significance Tests**: Verification of result reliability

This dual-track research approach will provide both scientific insights about individual techniques and practical guidance on how to best combine approaches for maximum effectiveness in real-world applications.
