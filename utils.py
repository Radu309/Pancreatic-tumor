import numpy as np
import os
import sys


data_path = sys.argv[1]

# Define paths at the module level
image_path = os.path.join(data_path, 'images')
mask_path = os.path.join(data_path, 'masks')
list_path = os.path.join(data_path, 'lists')
model_path = os.path.join(data_path, 'models')
log_path = os.path.join(data_path, 'logs')
dataset_path = os.path.join(data_path, 'dataset')

# Ensure directories exist
paths = [image_path, mask_path, list_path, model_path, log_path, dataset_path]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

image_path_ = {plane: os.path.join(data_path, 'images_' + plane) for plane in ['Z']}
mask_path_ = {plane: os.path.join(data_path, 'masks_' + plane) for plane in ['Z']}
list_training = {plane: os.path.join(data_path, 'training_' + plane + '.txt') for plane in ['Z']}

for path in image_path_.values():
    if not os.path.exists(path):
        os.makedirs(path)

for path in mask_path_.values():
    if not os.path.exists(path):
        os.makedirs(path)


def preprocess(images):
    """add one more axis as tf require"""
    images = images[..., np.newaxis]
    return images


def preprocess_front(images):
    images = images[np.newaxis, ...]
    return images


# returning the binary mask map by the organ ID (especially useful under overlapping cases)
#   mask: the mask matrix
#   organ_ID: the organ ID
def is_organ(mask, organ_ID):
    return mask == organ_ID


def pad_2d(image, plane, padval, xmax, ymax, zmax):
    if plane == 'X':
        npad = ((0, ymax - image.shape[1]), (0, zmax - image.shape[2]))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values=padval)
    elif plane == 'Z':
        npad = ((0, xmax - image.shape[0]), (0, ymax - image.shape[1]))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values=padval)
    return padded


def in_training_set(total_samples, i, folds, current_fold):
    fold_remainder = folds - total_samples % folds
    fold_size = (total_samples - total_samples % folds) / folds
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (start_index <= i < end_index)


# returning the filename of the training set according to the current fold ID
def training_set_filename(current_fold):
    return os.path.join(list_path, 'training_' + 'FD' + str(current_fold) + '.txt')


# returning the filename of the testing set according to the current fold ID
def testing_set_filename(current_fold):
    return os.path.join(list_path, 'testing_' + 'FD' + str(current_fold) + '.txt')


# computing the DSC together with other values based on the mask and prediction volumes
def DSC_computation(mask, pred):
    pred_sum = pred.sum()
    mask_sum = mask.sum()
    inter_sum = np.logical_and(pred, mask).sum()
    return 2 * float(inter_sum) / (pred_sum + mask_sum), inter_sum, pred_sum, mask_sum
