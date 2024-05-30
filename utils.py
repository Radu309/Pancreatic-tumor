import sys

import numpy as np
import os
from sklearn.metrics import jaccard_score

DATA_PATH = "data/Pancreas_Segmentation"

# Define paths at the module level
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
IMAGE_PATH = os.path.join(DATASET_PATH, 'images')
MASK_PATH = os.path.join(DATASET_PATH, 'masks')
IMAGE_NPY_PATH = os.path.join(DATASET_PATH, 'NPY_Images')
MASK_NPY_PATH = os.path.join(DATASET_PATH, 'NPY_Masks')
IMAGE_CT_PATH = os.path.join(DATASET_PATH, 'CT_Images')
MASK_CT_PATH = os.path.join(DATASET_PATH, 'CT_Masks')

DATALOADER_PATH = os.path.join(DATA_PATH, 'dataloader')
TRAIN_DATALOADER_PATH = os.path.join(DATALOADER_PATH, 'train')
TEST_DATALOADER_PATH = os.path.join(DATALOADER_PATH, 'test')

LISTS_PATH = os.path.join(DATA_PATH, 'lists')
MODELS_PATH = os.path.join(DATA_PATH, 'models')
METRICS_PATH = os.path.join(DATA_PATH, 'metrics')
PREDICTED_PATH = os.path.join(DATA_PATH, 'predicted')

paths = [DATASET_PATH, IMAGE_PATH, MASK_PATH, IMAGE_NPY_PATH, MASK_NPY_PATH, DATALOADER_PATH,
         TRAIN_DATALOADER_PATH, TEST_DATALOADER_PATH, LISTS_PATH, MODELS_PATH, METRICS_PATH, PREDICTED_PATH]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

LIST_DATASET = os.path.join(LISTS_PATH, 'dataset' + '.txt')


# returning the filename of the training set according to the current fold ID
def training_set_filename(file_sliced, total):
    return os.path.join(LISTS_PATH, f'training_{file_sliced}_of_{total}.txt')


# returning the filename of the testing set according to the current fold ID
def testing_set_filename(file_sliced, total):
    return os.path.join(LISTS_PATH, f'testing_{file_sliced}_of_{total}.txt')


def in_training_set(total_samples, i, slices, current_fold):
    fold_remainder = slices - total_samples % slices
    fold_size = (total_samples - total_samples % slices) / slices
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (start_index <= i < end_index)


def normalize_image(image, low_range, high_range):
    return (image - low_range) / float(high_range - low_range)


def pad_2d(image, pad_val, xmax, ymax):
    val = ((0, xmax - image.shape[0]), (0, ymax - image.shape[1]))
    padded = np.pad(image, pad_width=val, mode='constant', constant_values=pad_val)
    return padded


def dice_coefficient(y_true, y_pred, smooth):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


# New function to calculate IoU
def iou_score(y_true, y_pred):
    y_true_f = y_true.view(-1).cpu().numpy()
    y_pred_f = y_pred.view(-1).detach().cpu().numpy()
    y_pred_f = (y_pred_f > 0.5).astype(np.uint8)
    return jaccard_score(y_true_f, y_pred_f)


def precision_score(y_true, y_pred):
    true_positives = (y_true * y_pred).sum()
    predicted_positives = y_pred.sum()
    precision = true_positives / (predicted_positives + 1e-8)  # Add small value to avoid division by zero
    return precision
