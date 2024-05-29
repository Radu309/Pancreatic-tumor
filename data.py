import torch
import numpy as np
import os
import sys
from utils import training_set_filename, testing_set_filename, normalize_image, pad_2d


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
        images_list = open(training_set_filename(lists_path, percent), 'r').read().splitlines()
    else:
        images_list = open(testing_set_filename(lists_path, percent), 'r').read().splitlines()

    image_set = np.zeros((len(images_list)), dtype=int)
    for i in range(len(images_list)):
        s = images_list[i].split(' ')
        image_set[i] = int(s[0])

    slice_list = open(list_dataset, 'r').read().splitlines()
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

        images_list_normalized[i] = pad_2d(cropped_image, 0, X_MAX, Y_MAX)
        masks_list_normalized[i] = pad_2d(cropped_mask, 0, X_MAX, Y_MAX)

        if i % 100 == 0:
            print(f'Done: {i}/{total} slices')

    if data_type == 'train':
        train_images = torch.tensor(images_list_normalized[:-val_count])
        train_masks = torch.tensor(masks_list_normalized[:-val_count])
        val_images = torch.tensor(images_list_normalized[-val_count:])
        val_masks = torch.tensor(masks_list_normalized[-val_count:])

        torch.save((train_images, train_masks, val_images, val_masks),
                   os.path.join(train_dataloader_path, f'train_{percent}%_LR-{low_range}_HR-{high_range}.pt'))
        print(f'Training data created for percent = {percent}')
    else:
        test_images = torch.tensor(images_list_normalized)
        test_masks = torch.tensor(masks_list_normalized)

        torch.save((test_images, test_masks),
                   os.path.join(test_dataloader_path, f'test_{100-percent}%_LR-{low_range}_HR-{high_range}.pt'))
        print(f'Testing data created for percent = {percent}')


def load_train_and_val_data(path, current_percent, lr, hr):
    train_images, train_masks, val_images, val_masks = (
        torch.load(os.path.join(path, f'train_{current_percent}%_LR-{lr}_HR-{hr}.pt')))
    train_dataset = PrepareDataset(train_images, train_masks)
    val_dataset = PrepareDataset(val_images, val_masks)
    return train_dataset, val_dataset


def load_test_data(path, current_percent, lr, hr):
    test_images, test_masks = (
        torch.load(os.path.join(path, f'test_{100-current_percent}%_LR-{lr}_HR-{hr}.pt')))
    test_dataset = PrepareDataset(test_images, test_masks)
    return test_dataset


if __name__ == '__main__':
    list_dataset = sys.argv[1]
    lists_path = sys.argv[2]
    train_dataloader_path = sys.argv[3]
    test_dataloader_path = sys.argv[4]

    percent = int(sys.argv[5])
    Z_MAX = int(sys.argv[6])
    Y_MAX = int(sys.argv[7])
    X_MAX = int(sys.argv[8])
    margin = int(sys.argv[9])
    low_range = int(sys.argv[10])
    high_range = int(sys.argv[11])

    create_data('train')
    create_data('test')
