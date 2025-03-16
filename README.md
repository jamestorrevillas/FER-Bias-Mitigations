# FER-BIAS-MITIGATIONS: Facial Emotion Recognition Bias Mitigation Strategies

This repository implements and compares different bias mitigation strategies for facial emotion recognition (FER) systems. It addresses demographic biases in facial analysis by examining different techniques to improve fairness across gender, age, and other protected attributes.

## Project Overview

Facial emotion recognition systems often exhibit biased performance across demographic groups. This project implements several approaches to mitigate these biases:

1. **Data Augmentation**: Targeted augmentation to balance emotion and demographic representations
2. **Dynamic Cross-Dataset Weighting**: Adaptive weighting strategy for multi-domain training
3. **Multi-Task Learning**: Joint training of emotion recognition with demographic attributes

The project uses a pre-trained FER+ model as the baseline and evaluates improvements on a limited version of the RAF-DB dataset with manually annotated demographic information.

## Dataset & Resources

⚠️ **Note:** Due to size constraints, dataset files and pre-trained models are not included in this repository.

### Download Required Resources

1. Download the resources files from [Google Drive Link](https://drive.google.com/drive/folders/1SNroKS9wUcT6sTGR8BuoPJN6a9DPSYRX?usp=sharing)
2. Extract the downloaded dataset ZIP file
3. Place the extracted `dataset` folder inside the `resources` directory of this project

The resources package includes:
- FER/FER+ dataset
- Limited RAF-DB dataset from Kaggle
- Pre-trained baseline model
- Augmented datasets

### Dataset Information

#### FER/FER+ Dataset
- **Description**: Facial Expression Recognition dataset with crowd-sourced emotion labels
- **Size**: 35,887 grayscale 48×48 pixel face images
- **Classes**: 8 emotion categories (neutral, happiness, surprise, sadness, anger, disgust, fear, contempt)
- **Source**: [FER+ GitHub Repository](https://github.com/microsoft/FERPlus)

#### RAF-DB Dataset (Limited Version)
- **Description**: Limited version of the Real-world Affective Faces Database from Kaggle
- **Size**: 15,339 facial images (12,271 training images and 3,068 test images)
- **Classes**: 7 basic emotion categories (surprise, fear, disgust, happiness, sadness, anger, neutral)
- **Demographics**: Manually annotated gender (male/female) and age group (0-12, 13-19, 20-40, 41-60, 60+) labels
- **Note**: We did not use the original and full RAF-DB dataset due to permission restrictions. Instead, we utilized a limited version available on Kaggle and manually annotated demographic attributes as the limited version only contained emotion labels.
- **Source**: [Kaggle RAF-DB Dataset](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)

### Model Information

#### baseline-ferplus-model.h5
- Based on the implementation from [Vick Sam's FER Model](https://github.com/vicksam/fer-model)
- Architecture: Custom CNN with separable convolutions
- Input: 48×48 grayscale images
- Output: 8 emotion classes (FER+ labels)
- Parameters: ~170K

#### Fine-tuned Models
The repository includes several fine-tuned models that build upon the baseline:
- **emotion-augmentation-finetuned-model.h5**: Fine-tuned using emotion-balanced augmented data
- **demographic-augmentation-finetuned-model.h5**: Fine-tuned using demographically-balanced data
- **multitask-finetuned-model.h5**: Multi-task learning with emotion and demographic outputs

### Resources Directory Structure

```
resources/
├── dataset/
│   ├── fer/
│   │   ├── augmented/              # Augmented FER+ dataset
│   │   │   ├── augmented_images.npy
│   │   │   ├── augmented_labels.npy
│   │   │   └── augmentation_stats/ # Statistics and visualizations
│   │   └── labels/                 # Original FER+ labels
│   │       ├── fer2013.csv
│   │       ├── fer2013new.csv
│   │       └── dataset.csv
│   │
│   └── raf-db/
│       ├── augmented/              # Augmented RAF-DB dataset
│       │   ├── augmented_images.npy
│       │   ├── augmented_labels.npy
│       │   └── augmented_demographics.pkl
│       ├── dataset/                # Limited RAF-DB images from Kaggle
│       │   ├── train/              # 12,271 images
│       │   └── test/               # 3,068 images
│       └── labels/                 # RAF-DB labels with manually added demographics
│           ├── train_labels.csv
│           └── test_labels.csv
│
└── models/
    ├── baseline-ferplus-model.h5
    ├── emotion-augmentation-finetuned-model.h5
    ├── demographic-augmentation-finetuned-model.h5
    └── multitask-finetuned-model.h5
```

## Installation

```bash
# Clone the repository
git clone https://github.com/jamestorrevillas/FER-Bias-Mitigations.git
cd FER-Bias-Mitigations

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (TensorFlow, NumPy, Pandas, Matplotlib, etc.)
# Note: See import statements in code files for specific requirements
```

## Project Structure

```
FER-BIAS-MITIGATIONS/
├── baseline_ferplus_model_training/   # Baseline model training code
├── bias_mitigation_approaches/        # Implementation of bias mitigation approaches
│   ├── data_augmentation/             # Data augmentation strategies
│   ├── dynamic_weighting/             # Dynamic cross-dataset weighting
│   └── multitask_learning/            # Multi-task learning with demographic attributes
├── resources/                         # Datasets and pre-trained models (download separately)
│   ├── dataset/                       # FER+, limited RAF-DB datasets
│   └── models/                        # Pre-trained models
└── utils/                             # Shared utility functions and analysis tools
    └── comparative_analysis/          # Tools for comparing different approaches
```

## Usage

### Running the Baseline Evaluation

```bash
python utils/comparative_analysis/analyze.py
```

To analyze different models, modify the `models_config` dictionary in `analyze.py`.

### Data Augmentation Approach

```bash
# Generate balanced augmentations
python bias_mitigation_approaches/data_augmentation/emotion_augmentation.py
python bias_mitigation_approaches/data_augmentation/demographic_augmentation.py

# Train with augmented data
python bias_mitigation_approaches/data_augmentation/emotion_augmentation.py
python bias_mitigation_approaches/data_augmentation/demographic_augmentation.py
```

### Multi-Task Learning Approach

```bash
# Run from project root to ensure proper imports
python -m bias_mitigation_approaches.multitask_learning.multitask_fairness
```

### Dynamic Cross-Dataset Weighting

```bash
# Run from project root to ensure proper imports
python -m bias_mitigation_approaches.dynamic_weighting.dynamic_weighting
```

### Comparative Analysis

```bash
python utils/comparative_analysis/analyze.py
```

## Results

The implementation compares different bias mitigation strategies based on:

- Overall accuracy on the test set
- Fairness metrics (demographic parity, equalized odds)
- Per-group accuracies across genders and age groups
- Emotion recognition accuracy by demographic intersections

Detailed results and visualizations are generated in the `results` directory when running the comparative analysis.

## Usage Notes

- The FER+ dataset uses 48×48 grayscale images with 8 emotion classes
- The limited RAF-DB dataset uses 100×100 color images with 7 emotion classes
- Both datasets have been normalized and preprocessed for compatibility
- The augmented datasets contain balanced samples to mitigate demographic bias
- The demographic attributes (age and gender) for the limited RAF-DB dataset were manually annotated since they were not included in the Kaggle version

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Research Team

- [James Torrevillas](https://github.com/jamestorrevillas)
- [Carl Gerard Resurreccion](https://github.com/carlgerardresurreccion)
- [Alyssa Vivien Cañas](https://github.com/Canas-AlyssaVivien)

### Dataset and Research Citations

#### FER+ Dataset
```
@inproceedings{BarsoumICMI2016,
  title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
  author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
  booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
  year={2016}
}
```

#### RAF-DB Dataset (Original)
```
@inproceedings{li2017reliable,
  title={Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild},
  author={Li, Shan and Deng, Weihong and Du, JunPing},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  pages={2584--2593},
  year={2017}
}
```

#### Bias in Facial Expression Recognition
```
@article{xu2020investigating,
  title={Investigating Bias and Fairness in Facial Expression Recognition},
  author={Xu, Tian and White, Jennifer and Kalkan, Sinan and Gunes, Hatice},
  journal={arXiv preprint arXiv:2007.10075},
  year={2020}
}
```

## Acknowledgements

This work builds upon the following datasets, repositories and research:

- FER and FER+ datasets
- Limited version of RAF-DB (Real-world Affective Faces Database) from Kaggle with manual demographic annotations
- [FER Model by Vick Sam](https://github.com/vicksam/fer-model) - Baseline implementation of facial emotion recognition
- Xu, T., White, J., Kalkan, S., & Gunes, H. (2020). [Investigating Bias and Fairness in Facial Expression Recognition](https://arxiv.org/abs/2007.10075). arXiv preprint arXiv:2007.10075.
