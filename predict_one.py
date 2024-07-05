import os

import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk_interface

from unet import UNet

import numpy as np
from utils import normalize_image, LIST_DATASET, pad_2d, dice_coefficient, iou_score, precision_score, recall_score, \
    specificity_score, f1_score

margin = 40
Y_MAX = 256
X_MAX = 192


def normalize_and_split_path(path):
    return os.path.normpath(path).split(os.sep)


def find_mask_path(image_path, LIST_DATASET):
    image_path_parts = normalize_and_split_path(image_path)
    with open(LIST_DATASET, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 2:
                dataset_image_path_parts = normalize_and_split_path(parts[2])
                if image_path_parts == dataset_image_path_parts:
                    return os.path.normpath(parts[3])
    return None


def load_single_image(image_path):
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

    return torch.tensor(cropped_image).unsqueeze(0), torch.tensor(current_mask).unsqueeze(0), torch.tensor(full_image).unsqueeze(0), minA, minB,


def predict_images(image_path, model_pancreas_path, model_tumor_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cropped_image, original_mask, original_image, minY, minX = load_single_image(image_path)
    cropped_image = cropped_image.to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
    original_image = original_image.to(device, dtype=torch.float32, non_blocking=True)
    original_mask = original_mask.to(device, dtype=torch.float32, non_blocking=True)

    model_pancreas = UNet(1, 1).to(device)
    model_pancreas.load_state_dict(torch.load(model_pancreas_path))
    model_pancreas.eval()

    model_tumor = UNet(1, 1).to(device)
    model_tumor.load_state_dict(torch.load(model_tumor_path))
    model_tumor.eval()

    with torch.no_grad():
        outputs_pancreas = model_pancreas(cropped_image)
        outputs_tumor = model_tumor(cropped_image)

    original_image_np = original_image.cpu().numpy().squeeze()
    original_mask_np = original_image.cpu().numpy().squeeze()
    cropped_image_np = cropped_image.cpu().numpy().squeeze()
    outputs_pancreas_np = outputs_pancreas.detach().cpu().numpy().squeeze()
    outputs_tumor_np = outputs_tumor.detach().cpu().numpy().squeeze()

    top_left = (minX - margin, minY - margin)
    h, w = cropped_image_np.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Create full-size padded masks
    padded_tumor_output = np.zeros_like(original_image_np)
    padded_pancreas_output = np.zeros_like(original_image_np)
    padded_tumor_output[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = outputs_tumor_np
    padded_pancreas_output[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = outputs_pancreas_np

    # Calculate metrics
    dice_tumor = dice_coefficient(original_mask, torch.tensor(padded_tumor_output).to(device)).item()
    iou_tumor = iou_score(original_mask, torch.tensor(padded_tumor_output).to(device))
    precision_tumor = precision_score(original_mask, torch.tensor(padded_tumor_output).to(device)).item()
    recall_tumor = recall_score(original_mask, torch.tensor(padded_tumor_output).to(device)).item()
    specificity_tumor = specificity_score(original_mask, torch.tensor(padded_tumor_output).to(device)).item()
    f1_tumor = f1_score(precision_tumor, recall_tumor)
    accuracy_tumor = ((torch.tensor(padded_tumor_output).to(device).round() == original_mask).float().mean().item())

    # Print metrics
    print(f"Tumor - Dice: {dice_tumor}, IoU: {iou_tumor}, Precision: {precision_tumor}, Recall: {recall_tumor}, Specificity: {specificity_tumor}, F1: {f1_tumor}, Accuracy: {accuracy_tumor}")

    show_output(original_image_np, padded_pancreas_output, padded_tumor_output)


def show_output(original_image, padded_pancreas_output, padded_tumor_output):
    overlay_rgb = np.stack([original_image] * 3, axis=-1)
    red_mask = padded_pancreas_output > 0.5
    blue_mask = padded_tumor_output > 0.5

    # Set red channel where the pancreas mask is true
    overlay_rgb[red_mask, 0] = 1  # Red
    overlay_rgb[red_mask, 1:] = 0  # Remove green and blue

    # Set blue channel where the tumor mask is true
    overlay_rgb[blue_mask, 2] = 1  # Blue
    overlay_rgb[blue_mask, :2] = 0  # Remove red and green

    # Create a new Tkinter window
    window = tk_interface.Tk()
    window.title('Image')

    # Create a figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(overlay_rgb)
    ax.set_title('CT image of a pancreatic tumor')

    # Create a frame to hold the canvas and the color description
    frame = tk_interface.Frame(window)
    frame.pack(fill=tk_interface.BOTH, expand=True)

    # Create a canvas to display the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk_interface.LEFT, fill=tk_interface.BOTH, expand=True)

    # Create a label to display the color description
    description_label = tk_interface.Label(frame, text="\nRoșu: Pancreas\nAlbastru: Tumoare",
                                           justify=tk_interface.LEFT, padx=10, pady=10)
    description_label.pack(side=tk_interface.RIGHT, fill=tk_interface.BOTH, expand=True)

    # Start the Tkinter main loop
    window.mainloop()



