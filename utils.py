import numpy as np
import os
import sys
from sklearn.metrics import jaccard_score

data_path = sys.argv[1]

# Define paths at the module level
image_path = os.path.join(data_path, 'images')
mask_path = os.path.join(data_path, 'masks')
list_path = os.path.join(data_path, 'lists')
dataset_path = os.path.join(data_path, 'dataset')
execution_path = os.path.join(data_path, 'executions')

# Ensure directories exist
paths = [image_path, mask_path, list_path, dataset_path, execution_path]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

list_training = os.path.join(data_path, 'training_Z' + '.txt')


def normalize_image(image, low_range, high_range):
    return (image - low_range) / float(high_range - low_range)


def pad_2d(image, pad_val, xmax, ymax):
    # for axial plane
    val = ((0, xmax - image.shape[0]), (0, ymax - image.shape[1]))
    padded = np.pad(image, pad_width=val, mode='constant', constant_values=pad_val)
    return padded


def in_training_set(total_samples, i, folds, current_fold):
    fold_remainder = folds - total_samples % folds
    fold_size = (total_samples - total_samples % folds) / folds
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (start_index <= i < end_index)


def dice_coefficient(y_true, y_pred, smooth):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2.0 * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


# returning the filename of the training set according to the current fold ID
def training_set_filename(current_fold):
    return os.path.join(list_path, 'training_fold_' + str(current_fold) + '.txt')


# returning the filename of the testing set according to the current fold ID
def testing_set_filename(current_fold):
    return os.path.join(list_path, 'testing_fold_' + str(current_fold) + '.txt')


# computing the DSC together with other values based on the mask and prediction volumes
def DSC_computation(mask, pred):
    pred_sum = pred.sum()
    mask_sum = mask.sum()
    inter_sum = np.logical_and(pred, mask).sum()
    return 2 * float(inter_sum) / (pred_sum + mask_sum), inter_sum, pred_sum, mask_sum


# New function to calculate IoU
def iou_score(y_true, y_pred):
    y_true_f = y_true.view(-1).cpu().numpy()
    y_pred_f = y_pred.view(-1).detach().cpu().numpy()
    y_pred_f = (y_pred_f > 0.5).astype(np.uint8)
    return jaccard_score(y_true_f, y_pred_f)


def get_next_execution_id():
    existing_ids = [int(d.split('_')[1]) for d in os.listdir(execution_path) if d.startswith('execution_')]
    if not existing_ids:
        return 1
    return max(existing_ids) + 1

