import os
import sys

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

from resnet import ResUNet
from unet import UNet

import numpy as np
from utils import normalize_image, LIST_DATASET, pad_2d, PREDICTED_ALL


class SliceDataset(Dataset):
    def __init__(self, cropped_images, images, masks, list_min_X, list_min_Y):
        self.cropped_images = cropped_images
        self.images = images
        self.masks = masks
        self.list_min_X = list_min_X
        self.list_min_Y = list_min_Y

    def __len__(self):
        return len(self.cropped_images)

    def __getitem__(self, idx):
        return self.cropped_images[idx], self.images[idx], self.masks[idx], self.list_min_X[idx], self.list_min_Y[idx]


def slice_data_loader(margin, Y_MAX, X_MAX):
    slice_list = open(LIST_DATASET, 'r').read().splitlines()
    slices = len(slice_list)
    image_ID = np.zeros(slices, dtype=int)
    image_filename = ['' for _ in range(slices)]
    mask_filename = ['' for _ in range(slices)]
    for i in range(slices):
        s = slice_list[i].split(' ')
        image_ID[i] = s[0]
        image_filename[i] = s[2]
        mask_filename[i] = s[3]

    create_slice_list = []
    create_mask_list = []
    for i in range(slices):
        create_slice_list.append(image_filename[i])
        create_mask_list.append(mask_filename[i])

    total = len(create_slice_list)

    cropped_images_list_normalized = np.ndarray((total, X_MAX, Y_MAX), dtype=np.float32)
    images_list_normalized = np.ndarray((total, 512, 512), dtype=np.float32)
    masks_list_normalized = np.ndarray((total, 512, 512), dtype=np.float32)
    list_min_A = []
    list_min_B = []
    for i in range(total):
        current_image = np.load(create_slice_list[i])
        current_mask = np.load(create_mask_list[i])

        current_image = normalize_image(current_image, np.min(current_image), np.max(current_image))

        if current_image.max() > 1:
            current_image = current_image / current_image.max()

        images_list_normalized[i] = current_image
        masks_list_normalized[i] = current_mask

        arr = np.nonzero(current_mask)

        width = current_mask.shape[0]
        height = current_mask.shape[1]

        minA = min(arr[0])
        maxA = max(arr[0])
        minB = min(arr[1])
        maxB = max(arr[1])

        list_min_A.append(minA)
        list_min_B.append(minB)
        cropped_image = current_image[max(minA - margin, 0): min(maxA + margin + 1, width), \
                        max(minB - margin, 0): min(maxB + margin + 1, height)]

        cropped_images_list_normalized[i] = pad_2d(cropped_image, 0, X_MAX, Y_MAX)
        if i % 10 == 0:
            print(f'Done: {i}/{total} slices')
    cropped_images = torch.tensor(cropped_images_list_normalized)
    images = torch.tensor(images_list_normalized)
    masks = torch.tensor(masks_list_normalized)
    return SliceDataset(cropped_images, images, masks, list_min_A, list_min_B)


def predict_images(model_pancreas_path, model_tumor_path, margin, Y_MAX, X_MAX):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = slice_data_loader(margin, Y_MAX, X_MAX)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model_pancreas = UNet(1, 1).to(device)
    model_pancreas.load_state_dict(torch.load(model_pancreas_path))
    model_pancreas.eval()

    # model_tumor = UNet(1, 1).to(device)
    model_tumor = ResUNet(1, 1).to(device)
    model_tumor.load_state_dict(torch.load(model_tumor_path))
    model_tumor.eval()

    print("\t\tStart predict")
    for idx, (cropped_images, images, masks, minA, minB) in enumerate(test_loader):
        cropped_images = cropped_images.to(device, dtype=torch.float32, non_blocking=True)
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device, dtype=torch.float32, non_blocking=True)
        cropped_images = cropped_images.unsqueeze(1)
        outputs_pancreas = model_pancreas(cropped_images)
        outputs_tumor = model_tumor(cropped_images)

        images_np = images.cpu().numpy().squeeze()
        masks_np = masks.cpu().numpy().squeeze()
        cropped_images_np = cropped_images.cpu().numpy().squeeze()
        outputs_pancreas_np = outputs_pancreas.detach().cpu().numpy().squeeze()
        outputs_tumor_np = outputs_tumor.detach().cpu().numpy().squeeze()

        original_image = images_np
        original_mask = masks_np
        cropped_image = cropped_images_np
        output_pancreas = outputs_pancreas_np
        output_tumor = outputs_tumor_np

        top_left = (minB.item() - margin, minA.item() - margin)
        h, w = cropped_image.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Create full-size padded masks
        padded_tumor_output = np.zeros_like(original_image)
        padded_pancreas_output = np.zeros_like(original_image)
        padded_tumor_output[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = output_tumor
        padded_pancreas_output[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = output_pancreas

        save_output(original_image, padded_pancreas_output, padded_tumor_output, original_mask, PREDICTED_ALL, idx)


def save_output(original_image, padded_pancreas_output, padded_tumor_output, mask, save_dir, idx):
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    im0 = ax[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title('Original Image')
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(padded_pancreas_output, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title('Pancreas Output')
    fig.colorbar(im1, ax=ax[1])

    im2 = ax[2].imshow(padded_tumor_output, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title('Tumor Output')
    fig.colorbar(im2, ax=ax[2])

    im3 = ax[3].imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax[3].set_title('Original Mask tumor')
    fig.colorbar(im3, ax=ax[3])

    plt.savefig(os.path.join(save_dir, f'{idx:04d}.jpg'))
    plt.close(fig)

    overlay_rgb = np.stack([original_image] * 3, axis=-1)
    red_mask = padded_pancreas_output > 0.5
    blue_mask = padded_tumor_output > 0.5

    # Set red channel where the pancreas mask is true
    overlay_rgb[red_mask, 0] = 1  # Red
    overlay_rgb[red_mask, 1:] = 0  # Remove green and blue

    # Set blue channel where the tumor mask is true
    overlay_rgb[blue_mask, 2] = 1  # Blue
    overlay_rgb[blue_mask, :2] = 0  # Remove red and green

    # Display the overlay image
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(overlay_rgb)
    ax.set_title('Overlay of Pancreas (Red) and Tumor (Blue) Outputs on Original Image')
    plt.savefig(os.path.join(save_dir, f'{idx:04d}_overlay.jpg'))
    plt.close(fig)


if __name__ == "__main__":
    model_pancreas_path = sys.argv[1]
    model_tumor_path = sys.argv[2]
    margin = 40
    Y_MAX = int(sys.argv[4])
    X_MAX = int(sys.argv[5])
    predict_images(model_pancreas_path, model_tumor_path, margin, Y_MAX, X_MAX)
    print("\t\tEnd predict")
