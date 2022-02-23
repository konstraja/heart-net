r"""
Custom PyTorch dataset that reads from the h5 datasets (see src.data module for more infos).
"""

import h5py
import random
import numpy as np
import torch
from src.common.image_util import stretched_image_from_skeleton_sequence
from src.common.joints import Joints

class Dataset(torch.utils.data.Dataset):
    r"""This custom PyTorch lazy loads from the h5 datasets. This means that it does not load the entire dataset in
    memory, which would be impossible for the IR sequences. Instead, it opens and reads from the h5 file. This is a bit
    slower, but very memory efficient. Additionally, the lost time is mitigated when using multiple workers for the
    data loaders.
    Attributes:
        - **data_path** (str): Path containing the h5 files (default *./data/processed/*).
        - **model_type** (str): "FUSION" only for now.
        - **use_pose** (bool): Include skeleton data
        - **use_ir** (bool): Include IR data
        - **use_cropped_IR** (bool): Type of IR dataset
        - **sub_sequence_length** (str): Number of frames to subsample from full IR sequences
        - **augment_data** (bool): Choose to augment data by geometric transformation (skeleton data) or horizontal
          flip (IR data)
        - **mirror_skeleton** (bool): Choose to perform mirroring on skeleton data (e.g. left hand becomes right hand)
        - **samples_names** (list): Contains the sequences names of the dataset (ie. train, validation, test)
        - **c_min** (float): Minimum coordinate after camera-subject normalization
        - **c_max** (float): Maximum coordinate after camera-subject normalization
    Methods:
        - *__getitem__(index)*: Returns the processed sequence (skeleton and/or IR) and its label
        - *__len__()*: Returns the number of elements in dataset.
    """
    def __init__(self,
                 data_path,
                 samples_names,
                 c_min = None,
                 c_max = None):
        super(Dataset, self).__init__()

        self.data_path = data_path

        self.samples_names = samples_names
        self.c_min = c_min
        self.c_max = c_max

        if c_max is None and c_min is None:
            print("Computing c_min and c_max. This takes a while ...")

            c_min = []
            c_max = []

            with h5py.File(self.data_path + "skeleton.h5", 'r') as skeleton_dataset:
                for sample_name in self.samples_names:
                    skeleton = skeleton_dataset[sample_name]["skeleton"][:]

                    # Perform normalization step
                    trans_vector = skeleton[:, 0, Joints.SPINEMID, :]  # shape (3, 2)
                    trans_vector[:, 1] = trans_vector[:, 0]
                    skeleton = (skeleton.transpose(1, 2, 0, 3) - trans_vector).transpose(2, 0, 1, 3)

                    # Update c_min and c_max
                    c_min.append(np.amin(skeleton))
                    c_max.append(np.amax(skeleton))

            self.c_min = np.amin(c_min)
            self.c_max = np.amax(c_max)
            print(f"Generated c_min {self.c_min} c_max {self.c_max}")

    def __getitem__(self, index):
        r"""Returns a processed sequence and label given an index.
        Inputs:
            - **index** (int): Used as an index for **samples_names** list which will yield a sequence
              name that will be used to address the h5 files.
        Outputs:
            - **skeleton_image** (np array): Skeleton sequence mapped to an image of shape `(3, 224, 224)`.
              Equals -1 if **use_pose** is False.
            - **ir_sequence** (np array): Subsampled IR sequence of shape `(sub_sequence_length, 112, 112)`.
              Equals -1 if **use_ir** is False.
            - **y** (int): Class label of sequence.
        """
        # Get label
        y = int(self.samples_names[index][-3:]) - 1

        # retrieve skeleton sequence of shape (3, max_frame, num_joint=25, 2)
        with h5py.File(self.data_path + "skeleton.h5", 'r') as skeleton_dataset:
            skeleton = skeleton_dataset[self.samples_names[index]]["skeleton"][:]

        # Potential outputs
        skeleton_image = -1

        # Normalize skeleton according to S-trans (see View Adaptive Network for details)
        # Subjects 1 and 2 have their own new coordinates system
        trans_vector = skeleton[:, 0, Joints.SPINEMID, :]

        # Subjects 1 and 2 are transposed into the coordinates system of subject 1
        trans_vector[:, 1] = trans_vector[:, 0]

        skeleton = (skeleton.transpose(1, 2, 0, 3) - trans_vector).transpose(2, 0, 1, 3)

        # shape (3, 224, 224)
        skeleton_image = np.float32(stretched_image_from_skeleton_sequence(skeleton, self.c_min, self.c_max))

        # Return corresponding data
        return [skeleton_image], y

    def __len__(self):
        r"""Returns number of elements in dataset
        Outputs:
            - **length** (int): Number of elements in dataset.
        """
        return len(self.samples_names)