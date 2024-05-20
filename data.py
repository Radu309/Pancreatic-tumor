import torch
import numpy as np
import os
import sys
from utils import *


class PrepareDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, mask_files, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def create_train_data(current_fold, plane):
    images_list = open(training_set_filename(current_fold), 'r').read().splitlines()
    training_image_set = np.zeros((len(images_list)), dtype=int)

    for i in range(len(images_list)):
        s = images_list[i].split(' ')
        training_image_set[i] = int(s[0])

    slice_list = open(list_training[plane], 'r').read().splitlines()
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
        pixels[i] = float(s[organ_ID * 5])

    create_slice_list = []
    create_mask_list = []

    for i in range(slices):
        if image_ID[i] in training_image_set and pixels[i] >= 100:
            create_slice_list.append(image_filename[i])
            create_mask_list.append(mask_filename[i])

    if len(create_slice_list) != len(create_mask_list):
        raise ValueError('slice number does not equal mask number!')

    dataset = PrepareDataset(create_slice_list, create_mask_list)
    torch.save(dataset, os.path.join(data_path, f'dataset/train_dataset_fold_{current_fold}_plane_{plane}.pt'))
    print(f'Training data created for fold {current_fold}, plane {plane}')


def load_train_data(fold, plane):
    return torch.load(os.path.join(data_path, f'dataset/train_dataset_fold_{fold}_plane_{plane}.pt'))


if __name__ == '__main__':
    data_path = sys.argv[1]
    folds = int(sys.argv[2])
    plane = sys.argv[3]

    ZMAX = int(sys.argv[4])
    YMAX = int(sys.argv[5])
    XMAX = int(sys.argv[6])

    margin = int(sys.argv[7])
    organ_ID = int(sys.argv[8])
    low_range = int(sys.argv[9])
    high_range = int(sys.argv[10])

    for current_fold in range(folds):
        create_train_data(current_fold, plane)

