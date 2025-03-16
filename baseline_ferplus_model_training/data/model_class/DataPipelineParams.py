# baseline_ferplus_model_training/data/model_class/DataPipelineParams.py

from enum import Enum


Augmentation = Enum('Augmentation', 'NONE MEDIUM HIGH')

Dataset = Enum('Dataset', 'FER FERPLUS')


class DataPipelineParams():
    '''Wrapper object for get_data_pipeline parameters'''

    def __init__(self,
                 dataset = Dataset.FER,
                 cross_entropy = False,
                 original_preprocessing = False,
                 batch_size = 32,
                 augmentation = Augmentation.NONE,
                 seed = 123,
                 use_augmented_data = False):  # New parameter
        '''Args:
            dataset(enum): based on it, labels of FER or FER-Plus will be used
            cross_entropy(boolean): whether labels should be class probabilities
                                    (has effect only for FER-Plus)
            original_preprocessing(boolean): whether apply original preprocessing
            batch_size(int)
            augmentation(enum): indicates level of augmentation to apply
            seed(int)
            use_augmented_data(boolean): whether to use the augmented dataset
        '''
        self.dataset = dataset
        self.cross_entropy = cross_entropy
        self.original_preprocessing = original_preprocessing
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.seed = seed
        self.use_augmented_data = use_augmented_data  # Store the flag