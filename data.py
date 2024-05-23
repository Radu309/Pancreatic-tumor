import torch
import numpy as np
import os
import sys
from utils import *
import matplotlib.pyplot as plt


class PrepareDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask


def create_train_data(current_fold):
    images_list = open(training_set_filename(current_fold), 'r').read().splitlines()
    training_image_set = np.zeros((len(images_list)), dtype=int)

    for i in range(len(images_list)):
        s = images_list[i].split(' ')
        training_image_set[i] = int(s[0])

    slice_list = open(list_training, 'r').read().splitlines()
    slices = len(slice_list)
    image_ID = np.zeros(slices, dtype=int)
    slice_ID = np.zeros(slices, dtype=int)
    image_filename = ['' for _ in range(slices)]
    mask_filename = ['' for _ in range(slices)]
    pixels = np.zeros(slices, dtype=int)

    for i in range(slices):
        s = slice_list[i].split(' ')
        image_ID[i] = s[0]
        slice_ID[i] = s[1]
        image_filename[i] = s[2]
        mask_filename[i] = s[3]
        pixels[i] = int(s[5])

    create_slice_list = []
    create_mask_list = []

    for i in range(slices):
        # check if the 2D image is in the other files by ID
        if image_ID[i] in training_image_set:
            create_slice_list.append(image_filename[i])
            create_mask_list.append(mask_filename[i])

    if len(create_slice_list) != len(create_mask_list):
        raise ValueError('slice number does not equal mask number!')

    total = len(create_slice_list)

    images_list_normalized = np.ndarray((total, 512, 512), dtype=np.float32)
    masks_list_normalized = np.ndarray((total, 512, 512), dtype=np.float32)
    for i in range(total):
        current_image = np.load(create_slice_list[i])
        current_mask = np.load(create_mask_list[i])

        # current_image = normalize_image(current_image, low_range, high_range)
        if current_image.min() != -100:
            print(f"Preprocessed Image {i} min: {current_image.min()}, max: {current_image.max()}")
        if current_mask.max() == 1:
            print(f"Preprocessed Image {i} min: {current_mask.min()}, max: {current_mask.max()}")

        if current_image.max() > 1:
            current_image = current_image / current_image.max()
        arr = np.nonzero(current_mask)

        width = current_mask.shape[0]
        height = current_mask.shape[1]

        minA = min(arr[0])
        maxA = max(arr[0])
        minB = min(arr[1])
        maxB = max(arr[1])

        cropped_image = current_image[max(minA - margin, 0): min(maxA + margin + 1, width),
                     max(minB - margin, 0): min(maxB + margin + 1, height)]
        cropped_mask = current_mask[max(minA - margin, 0): min(maxA + margin + 1, width),
                       max(minB - margin, 0): min(maxB + margin + 1, height)]

        # images_list_normalized[i] = pad_2d(cropped_image, 0, XMAX, YMAX)
        # masks_list_normalized[i] = pad_2d(cropped_mask, 0, XMAX, YMAX)

        images_list_normalized[i] = current_image
        masks_list_normalized[i] = current_mask

        if i % 100 == 0:
            print(f'Done: {i}/{total} slices')

    torch.save((torch.tensor(images_list_normalized), torch.tensor(masks_list_normalized)),
               os.path.join(data_path, f'dataset/train_dataset_fold_{current_fold}_plane_Z.pt'))

    print(f'Training data created for fold {current_fold}, plane Z')


def load_train_data(fold):
    images, masks = torch.load(os.path.join(data_path, f'dataset/train_dataset_fold_{fold}_plane_Z.pt'))
    return PrepareDataset(images, masks)


if __name__ == '__main__':
    data_path = sys.argv[1]
    folds = int(sys.argv[2])

    ZMAX = int(sys.argv[3])
    YMAX = int(sys.argv[4])
    XMAX = int(sys.argv[5])

    margin = int(sys.argv[6])
    low_range = int(sys.argv[7])
    high_range = int(sys.argv[8])

    for current_fold in range(folds):
        create_train_data(current_fold)

