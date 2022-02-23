import cv2
import numpy as np


def stretched_image_from_skeleton_sequence(skeleton, c_min, c_max):
    r"""Rotates the skeleton sequence around its different axis.
    Inputs:
        - **skeleton** (np array): Skeleton sequence of shape `(3 {x, y, z}, max_frame, num_joint=25, n_subjects=2)`
        - **c_min** (int): Minimum coordinate value across all sequences, joints, subjects, frames after the prior
          normalization step.
        - **c_max** (int): Maximum coordinate value across all sequences, joints, subjects, frames after the prior
          normalization step.
    Outputs:
        **skeleton_image** (np array): RGB image of shape `(3, 224, 224)`
    """

    max_frame = skeleton.shape[1]
    n_joints = skeleton.shape[2]

    # Reshape skeleton coordinates into an image
    skeleton_image = np.zeros((3, max_frame, 2 * n_joints))
    skeleton_image[:, :, 0:n_joints] = skeleton[:, :, :, 0]
    skeleton_image[:, :, n_joints:2 * n_joints] = skeleton[:, :, :, 1]
    skeleton_image = np.transpose(skeleton_image, (0, 2, 1))

    # Normalize (min-max)
    skeleton_image = np.floor(255 * (skeleton_image - c_min) / (c_max - c_min))  # shape (3, 2 * n_joints, max_frame)

    # Reshape image for ResNet
    skeleton_image = cv2.resize(skeleton_image.transpose(1, 2, 0), dsize=(224, 224)).transpose(2, 0, 1)

    return skeleton_image