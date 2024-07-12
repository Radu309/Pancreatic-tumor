import glob
import sys

import numpy as np
import os

import torch
from sklearn.metrics import jaccard_score

#Define the path to the dataset
# DATA_PATH = "data/Pancreas_Segmentation"
DATA_PATH = "data/Pancreas_Tumor_Segmentation"

# Define paths at the module level
DATASET_PATH = os.path.join(DATA_PATH, 'dataset')
IMAGE_PATH = os.path.join(DATASET_PATH, 'images')
MASK_PATH = os.path.join(DATASET_PATH, 'masks')
IMAGE_NPY_PATH = os.path.join(DATASET_PATH, 'NPY_Images')
MASK_NPY_PATH = os.path.join(DATASET_PATH, 'NPY_Masks')
IMAGE_CT_PATH = os.path.join(DATASET_PATH, 'CT_Images')
MASK_CT_PATH = os.path.join(DATASET_PATH, 'CT_Masks')

DATALOADER_PATH = os.path.join(DATA_PATH, 'dataloader')
LISTS_PATH = os.path.join(DATA_PATH, 'lists')
MODELS_PATH = os.path.join(DATA_PATH, 'models')
METRICS_PATH = os.path.join(DATA_PATH, 'metrics')
PREDICTED_PATH = os.path.join(DATA_PATH, 'predicted')

paths = [DATASET_PATH, IMAGE_PATH, MASK_PATH, IMAGE_NPY_PATH, MASK_NPY_PATH, DATALOADER_PATH,
         LISTS_PATH, MODELS_PATH, METRICS_PATH, PREDICTED_PATH]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

LIST_DATASET = os.path.join(LISTS_PATH, 'dataset' + '.txt')


# returning the filename of the training set according to the current fold ID
def training_set_filename(slices):
    return os.path.join(LISTS_PATH, f'training_{slices-1}_of_{slices}.txt')


# returning the filename of the testing set according to the current fold ID
def testing_set_filename(slices):
    return os.path.join(LISTS_PATH, f'testing_1_of_{slices}.txt')


def in_training_set(total_samples, i, slices, current_fold):
    fold_remainder = slices - total_samples % slices
    fold_size = (total_samples - total_samples % slices) / slices
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (start_index <= i < end_index)


def normalize_image(image, low_range, high_range):
    return (image - low_range) / float(high_range - low_range)


def pad_2d(image, pad_val, xmax, ymax):
    val = ((0, (xmax - image.shape[0])), (0, (ymax - image.shape[1])))
    padded = np.pad(image, pad_width=val, mode='constant', constant_values=pad_val)
    return padded


def dice_coefficient(y_true, y_pred, smooth=1e-4):
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


def recall_score(true_mask, pred_mask):
    tp = (true_mask * pred_mask).sum().float()
    fn = (true_mask * (1 - pred_mask)).sum().float()
    return tp / (tp + fn + 1e-10)


def specificity_score(true_mask, pred_mask):
    tn = ((1 - true_mask) * (1 - pred_mask)).sum().float()
    fp = ((1 - true_mask) * pred_mask).sum().float()
    return tn / (tn + fp + 1e-10)


def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall + 1e-10)


def precision_score(true_mask, pred_mask):
    # true_mask = torch.from_numpy(true_mask)
    # pred_mask = torch.from_numpy(pred_mask)
    true_mask = true_mask.float()  # Ensure the mask is a float tensor
    pred_mask = pred_mask.float()  # Ensure the predicted mask is a float tensor
    tp = (true_mask * pred_mask).sum().float()
    fp = ((1 - true_mask) * pred_mask).sum().float()
    return tp / (tp + fp + 1e-10)
