import numpy as np
import os
from sklearn.metrics import jaccard_score


# returning the filename of the training set according to the current fold ID
def training_set_filename(lists_path, current_fold):
    return os.path.join(lists_path, 'training_fold_' + str(current_fold) + '.txt')


# returning the filename of the testing set according to the current fold ID
def testing_set_filename(lists_path, current_fold):
    return os.path.join(lists_path, 'testing_fold_' + str(current_fold) + '.txt')


def in_training_set(total_samples, i, folds, current_fold):
    fold_remainder = folds - total_samples % folds
    fold_size = (total_samples - total_samples % folds) / folds
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


