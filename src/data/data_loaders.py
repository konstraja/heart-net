from asyncio.log import logger
import random
from typing import List
from torch.utils.data import DataLoader

from src.common.constants import XSUB_PERSONS
from src.data.dataset import Dataset

def get_data_loaders(data_path: str, sample_path: str, batch_size: int):
    training_samples, validation_samples, test_samples = split_data_set(sample_path)
    logger.info(f"Training samples: {len(training_samples)}, Validation samples: {len(validation_samples)}, Test samples: {len(test_samples)}")

    training_set = Dataset(data_path, training_samples)
    training_generator = DataLoader(training_set,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         pin_memory=True,
                                         num_workers=8)

    validation_set = Dataset(data_path, validation_samples,
                                  training_set.c_min,
                                  training_set.c_max)
    validation_generator = DataLoader(validation_set,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=8)

                                           
    testing_set = Dataset(data_path,
                               test_samples,
                               training_set.c_min,
                               training_set.c_max)

    test_generator = DataLoader(testing_set,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=8)
    return training_generator, validation_generator, test_generator

def split_data_set(sample_path: str, validation_ratio: float = 0.05) -> List:
    
    sample_names = [line.rstrip('\n') for line in open(sample_path)]
    person_suffixes = [ f"P{person_id:03d}" for person_id in XSUB_PERSONS]
    
    training_samples = [sample for sample in sample_names if any(xs in sample for xs in person_suffixes)]
    testing_samples = list(set(sample_names) - set(training_samples))

    training_samples = training_samples.copy()
    testing_samples = testing_samples.copy()

    validation_samples = [training_samples.pop(random.randrange(len(training_samples))) for _ in
                          range(int(validation_ratio * len(training_samples)))]

    return training_samples, validation_samples, testing_samples