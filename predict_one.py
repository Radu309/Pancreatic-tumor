import os
import sys

import cv2
import torch
from matplotlib import pyplot as plt
from unet import UNet

import numpy as np
from utils import normalize_image, LIST_DATASET, pad_2d, PREDICTED_ONE


def find_mask_path(image_path, LIST_DATASET):
    with open(LIST_DATASET, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts[2] == image_path:
                return parts[3]
    return None


def load_single_image(image_path, margin, Y_MAX, X_MAX):
    mask_path = find_mask_path(image_path, LIST_DATASET)
    if mask_path == None:
        exit("No mask found")

    current_image = np.load(image_path)
    current_mask = np.load(mask_path)

    current_image = normalize_image(current_image, np.min(current_image), np.max(current_image))

    if current_image.max() > 1:
        current_image = current_image / current_image.max()

    full_image = current_image

    arr = np.nonzero(current_mask)

    width = current_mask.shape[0]
    height = current_mask.shape[1]

    minA = min(arr[0])
    maxA = max(arr[0])
    minB = min(arr[1])
    maxB = max(arr[1])

    cropped_image = current_image[max(minA - margin, 0): min(maxA + margin + 1, width), \
                    max(minB - margin, 0): min(maxB + margin + 1, height)]

    cropped_image = pad_2d(cropped_image, 0, X_MAX, Y_MAX)

    return torch.tensor(cropped_image).unsqueeze(0), torch.tensor(full_image).unsqueeze(0)


def predict_images(image_path, model_pancreas_path, model_tumor_path, margin, Y_MAX, X_MAX):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cropped_image, original_image = load_single_image(image_path, margin, Y_MAX, X_MAX)
    cropped_image = cropped_image.to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
    original_image = original_image.to(device, dtype=torch.float32, non_blocking=True)

    model_pancreas = UNet(1, 1).to(device)
    model_pancreas.load_state_dict(torch.load(model_pancreas_path))
    model_pancreas.eval()

    model_tumor = UNet(1, 1).to(device)
    model_tumor.load_state_dict(torch.load(model_tumor_path))
    model_tumor.eval()

    print("\t\tStart predict")
    with torch.no_grad():
        outputs_pancreas = model_pancreas(cropped_image)
        outputs_tumor = model_tumor(cropped_image)

    original_image_np = original_image.cpu().numpy().squeeze()
    cropped_image_np = cropped_image.cpu().numpy().squeeze()
    outputs_pancreas_np = outputs_pancreas.detach().cpu().numpy().squeeze()
    outputs_tumor_np = outputs_tumor.detach().cpu().numpy().squeeze()

    result = cv2.matchTemplate(original_image_np, cropped_image_np, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = cropped_image_np.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Create full-size padded masks
    padded_tumor_output = np.zeros_like(original_image_np)
    padded_pancreas_output = np.zeros_like(original_image_np)
    padded_tumor_output[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = outputs_tumor_np
    padded_pancreas_output[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = outputs_pancreas_np

    save_output(original_image_np, padded_pancreas_output, padded_tumor_output, PREDICTED_ONE, 0)



def save_output(original_image, padded_pancreas_output, padded_tumor_output, save_dir, idx):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    im0 = ax[0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title('Original Image')
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(padded_pancreas_output, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title('Pancreas Output')
    fig.colorbar(im1, ax=ax[1])

    im2 = ax[2].imshow(padded_tumor_output, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title('Tumor Output')
    fig.colorbar(im2, ax=ax[2])

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
    margin = int(sys.argv[3])
    Y_MAX = int(sys.argv[4])
    X_MAX = int(sys.argv[5])
    current_image_path = sys.argv[6]
    predict_images(current_image_path, model_pancreas_path, model_tumor_path, margin, Y_MAX, X_MAX)
