import numpy as np
import os
from sklearn.metrics import jaccard_score


# returning the filename of the training set according to the current fold ID
def training_set_filename(lists_path, percent):
    return os.path.join(lists_path, 'training_' + str(percent) + '%_of_data' + '.txt')


# returning the filename of the testing set according to the current fold ID
def testing_set_filename(lists_path, percent):
    return os.path.join(lists_path, 'testing_' + str(100-percent) + '%_of_data' + '.txt')


def in_training_set(total_samples, i, percent):
    if percent < 0 or percent > 100:
        raise ValueError("Percentage must be between 0 and 100.")

    training_threshold = (total_samples * percent) / 100
    return 0 <= i < training_threshold


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
