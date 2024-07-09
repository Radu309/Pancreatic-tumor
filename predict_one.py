import os
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np
from unet import UNet
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
            if len(parts) > 2 and image_path_parts == normalize_and_split_path(parts[2]):
                return os.path.normpath(parts[3])
    return None


def load_single_image(image_path):
    mask_path = find_mask_path(image_path, LIST_DATASET)
    if mask_path is None:
        exit("No mask found")
    current_image = np.load(image_path)
    current_image = normalize_image(current_image, np.min(current_image), np.max(current_image))
    current_image = current_image / current_image.max() if current_image.max() > 1 else current_image
    current_mask = np.load(mask_path)

    arr = np.nonzero(current_mask)
    minA, maxA, minB, maxB = min(arr[0]), max(arr[0]), min(arr[1]), max(arr[1])
    cropped_image = pad_2d(current_image[max(minA - margin, 0): min(maxA + margin + 1, current_mask.shape[0]), \
                           max(minB - margin, 0): min(maxB + margin + 1, current_mask.shape[1])], 0, X_MAX, Y_MAX)

    return torch.tensor(cropped_image).unsqueeze(0), torch.tensor(current_mask).unsqueeze(0), torch.tensor(
        current_image).unsqueeze(0), minA, minB


def predict_images(image_path, unet_pancreas, unet_tumor, resnet_tumor, attention_tumor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cropped_image, original_mask, original_image, minY, minX = load_single_image(image_path)
    cropped_image, original_image, original_mask = [x.to(device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
                                                    for x in (cropped_image, original_image, original_mask)]

    with torch.no_grad():
        outputs = {name: model(cropped_image) for name, model in zip(
            ["pancreas", "unet_tumor", "resnet_tumor", "attention_tumor"],
            [unet_pancreas, unet_tumor, resnet_tumor, attention_tumor]
        )}

    def tensor_to_np(tensor):
        return tensor.detach().cpu().numpy().squeeze()

    outputs_np = {name: tensor_to_np(outputs[name]) for name in outputs}
    original_image_np = tensor_to_np(original_image)
    top_left = (minX - margin, minY - margin)
    bottom_right = (top_left[0] + cropped_image.shape[-1], top_left[1] + cropped_image.shape[-2])

    def create_padded_output(output):
        padded_output = np.zeros_like(original_image_np)
        padded_output[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = output
        return padded_output

    padded_outputs = {name: create_padded_output(outputs_np[name]) for name in outputs_np}

    def calculate_metrics(output_name):
        output_tensor = torch.tensor(padded_outputs[output_name]).to(device)
        return {
            "Dice": dice_coefficient(original_mask, output_tensor).item(),
            "IoU": iou_score(original_mask, output_tensor),
            "Precision": precision_score(original_mask, output_tensor).item(),
            "Recall": recall_score(original_mask, output_tensor).item(),
            "F1": f1_score(precision_score(original_mask, output_tensor).item(),
                           recall_score(original_mask, output_tensor).item()),
            "Accuracy": (output_tensor.round() == original_mask).float().mean().item()
        }

    metrics = {name: calculate_metrics(name) for name in ["unet_tumor", "resnet_tumor", "attention_tumor"]}

    def format_metrics(metrics):
        return "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()])

    show_output(original_image_np, padded_outputs["pancreas"], padded_outputs["unet_tumor"],
                padded_outputs["resnet_tumor"], padded_outputs["attention_tumor"],
                format_metrics(metrics["unet_tumor"]), format_metrics(metrics["resnet_tumor"]),
                format_metrics(metrics["attention_tumor"]))


def show_output(original_image, padded_pancreas_output, padded_unet_tumor_output, padded_resnet_tumor_output,
                padded_attention_tumor_output, unet_metrics_text, resnet_metrics_text, attention_metrics_text):
    def create_overlay(image, pancreas_output, tumor_output):
        overlay_rgb = np.stack([image] * 3, axis=-1)
        overlay_rgb[pancreas_output > 0.5, 0] = 1
        overlay_rgb[pancreas_output > 0.5, 1:] = 0
        overlay_rgb[tumor_output > 0.5, :2] = 0
        overlay_rgb[tumor_output > 0.5, 2] = 1
        return overlay_rgb

    images = [
        (padded_pancreas_output, padded_unet_tumor_output, 'CT image with U-Net tumor tumor', unet_metrics_text),
        (padded_pancreas_output, padded_resnet_tumor_output, 'CT image with ResUNet tumor output', resnet_metrics_text),
        (padded_pancreas_output, padded_attention_tumor_output, 'CT image with Attention U-Net tumor output',
         attention_metrics_text)
    ]

    window = tk.Tk()
    window.title('Images')
    window.geometry("1600x800")
    frame = tk.Frame(window)
    frame.pack(fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (pancreas_output, tumor_output, title, metrics_text) in zip(axes, images):
        overlay_rgb = create_overlay(original_image, pancreas_output, tumor_output)
        ax.imshow(overlay_rgb)
        ax.set_title(title)
        ax.axis('off')
        ax.text(0.5, -0.2, metrics_text, ha='center', va='top', transform=ax.transAxes, fontsize=10, wrap=True)

    canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    canvas_agg.draw()
    canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    window.mainloop()
