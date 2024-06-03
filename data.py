import torch
import numpy as np
import os
import sys
from utils import *


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


def create_data(data_type):
    if data_type == 'train':
        images_list = open(training_set_filename(slice_total), 'r').read().splitlines()
    else:
        images_list = open(testing_set_filename(slice_total), 'r').read().splitlines()

    image_set = np.zeros((len(images_list)), dtype=int)
    for i in range(len(images_list)):
        s = images_list[i].split(' ')
        image_set[i] = int(s[0])

    slice_list = open(LIST_DATASET, 'r').read().splitlines()
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
        if image_ID[i] in image_set:
            create_slice_list.append(image_filename[i])
            create_mask_list.append(mask_filename[i])

    if len(create_slice_list) != len(create_mask_list):
        raise ValueError('slice number does not equal mask number!')

    total = len(create_slice_list)
    val_count = int(total * 0.1)

    images_list_normalized = np.ndarray((total, X_MAX, Y_MAX), dtype=np.float32)
    masks_list_normalized = np.ndarray((total, X_MAX, Y_MAX), dtype=np.float32)

    for i in range(total):
        current_image = np.load(create_slice_list[i])
        current_mask = np.load(create_mask_list[i])

        current_image = normalize_image(current_image, low_range, high_range)

        if current_image.max() > 1:
            current_image = current_image / current_image.max()

        cropped_image = crop_image(current_image)
        cropped_mask = crop_image(current_mask)

        images_list_normalized[i] = cropped_image
        masks_list_normalized[i] = cropped_mask

        if i % 100 == 0:
            print(f'Done: {i}/{total} slices')

    if data_type == 'train':
        train_images = torch.tensor(images_list_normalized[:-val_count])
        train_masks = torch.tensor(masks_list_normalized[:-val_count])
        val_images = torch.tensor(images_list_normalized[-val_count:])
        val_masks = torch.tensor(masks_list_normalized[-val_count:])
        # train_images = images_list_normalized[:-val_count]
        # train_masks = masks_list_normalized[:-val_count]
        # val_images = images_list_normalized[-val_count:]
        # val_masks = masks_list_normalized[-val_count:]
        # np.save(os.path.join(DATALOADER_PATH, f'training_{slice_total-1}_of_{slice_total}_images.npy'), train_images)
        # np.save(os.path.join(DATALOADER_PATH, f'training_{slice_total-1}_of_{slice_total}_masks.npy'), train_masks)
        # np.save(os.path.join(DATALOADER_PATH, f'validation_{slice_total-1}_of_{slice_total}_images.npy'), val_images)
        # np.save(os.path.join(DATALOADER_PATH, f'validation_{slice_total-1}_of_{slice_total}_masks.npy'), val_masks)

        torch.save((train_images, train_masks, val_images, val_masks),
                   os.path.join(DATALOADER_PATH, f'training_{slice_total-1}_of_{slice_total}.pt'))
        print(f'Training data created for slice file number = {slice_total-1}_of_{slice_total}')
    else:
        # np.save(os.path.join(DATALOADER_PATH, f'testing_1_of_{slice_total}_images.npy'), images_list_normalized)
        # np.save(os.path.join(DATALOADER_PATH, f'testing_1_of_{slice_total}_masks.npy'), masks_list_normalized)
        test_images = torch.tensor(images_list_normalized)
        test_masks = torch.tensor(masks_list_normalized)

        torch.save((test_images, test_masks),
                   os.path.join(DATALOADER_PATH, f'testing_1_of_{slice_total}.pt'))
        print(f'Testing data created for slice file number = 1_of_{slice_total}')



def load_train_and_val_data(slice_total_):
    train_images, train_masks, val_images, val_masks = (
        torch.load(os.path.join(DATALOADER_PATH, f'training_{slice_total_-1}_of_{slice_total_}.pt')))
    train_dataset = PrepareDataset(train_images, train_masks)
    val_dataset = PrepareDataset(val_images, val_masks)
    return train_dataset, val_dataset


def load_test_data(slice_total_):
    test_images, test_masks = (
        torch.load(os.path.join(DATALOADER_PATH, f'testing_1_of_{slice_total_}.pt')))
    test_dataset = PrepareDataset(test_images, test_masks)
    return test_dataset


if __name__ == '__main__':
    slice_total = int(sys.argv[1])
    Z_MAX = int(sys.argv[2])
    Y_MAX = int(sys.argv[3])
    X_MAX = int(sys.argv[4])
    low_range = int(sys.argv[5])
    high_range = int(sys.argv[6])

    # for index in range(slice_total):
    #     slice_file = index + 1
    create_data('train')
    create_data('test')
