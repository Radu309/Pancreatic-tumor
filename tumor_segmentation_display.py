import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch

from attention import AttentionUNet
from predict_one import predict_images
from resnet import ResUNet
from unet import UNet
from utils import normalize_image


class TumorSegmentationDisplay:
    def __init__(self, root):
        self.root = root
        self.root.title("Tumor Grade Image Display")

        self.frame_width = 1070
        self.frame_height = 500
        self.root.geometry(f"{self.frame_width}x{self.frame_height}")

        self.image_figsize = (2, 2)
        self.margin = 10
        self.images_per_row = 5

        self.images = []
        self.current_index = 0

        self.main_frame = tk.Frame(root, width=self.frame_width, height=self.frame_height)
        self.main_frame.pack_propagate(False)
        self.main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_frame)
        self.scroll_y = tk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_frame = tk.Frame(self.canvas)

        self.scroll_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_y.pack(side="right", fill="y")

        self.load_images()
        self.display_images()

    def load_images(self):
        if IMAGES_PATH and os.path.isdir(IMAGES_PATH):
            for director, _, files in os.walk(IMAGES_PATH):
                directory_name = os.path.basename(director)
                for filename in files:
                    if filename.endswith('.npy'):
                        image_path = os.path.join(director, filename)
                        image_array = np.load(image_path)
                        image_normalized = normalize_image(image_array, np.min(image_array), np.max(image_array))
                        self.images.append((image_normalized, directory_name, image_path))

    def display_images(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        max_images_per_row = self.images_per_row
        num_images = min(self.current_index + 10, len(self.images)) - self.current_index
        rows = (num_images + max_images_per_row - 1) // max_images_per_row

        for row in range(rows):
            for col in range(max_images_per_row):
                idx = self.current_index + row * max_images_per_row + col
                if idx < len(self.images):
                    image, directory_name, image_path = self.images[idx]
                    fig, ax = plt.subplots(figsize=self.image_figsize)
                    ax.imshow(image, cmap='gray')
                    ax.set_title(directory_name)
                    ax.axis('off')
                    canvas = FigureCanvasTkAgg(fig, master=self.scroll_frame)
                    widget = canvas.get_tk_widget()
                    widget.grid(row=row, column=col, padx=self.margin // 2, pady=self.margin // 2)
                    widget.bind("<Button-1>", lambda e, path=image_path: self.on_image_click(e, path))
                    canvas.draw()

        next_button_frame = tk.Frame(self.scroll_frame)
        next_button_frame.grid(row=rows, column=max_images_per_row-1, sticky='se', padx=10, pady=10)
        back_button_frame = tk.Frame(self.scroll_frame)
        back_button_frame.grid(row=rows, column=0, sticky='se', padx=10, pady=10)

        if self.current_index > 0:
            back_button = tk.Button(back_button_frame, text="Back", command=self.display_previous_images)
            back_button.pack(side=tk.LEFT, padx=5)

        if len(self.images) > self.current_index + 10:
            next_button = tk.Button(next_button_frame, text="Next", command=self.display_next_images)
            next_button.pack(side=tk.LEFT, padx=5)

    def on_image_click(self, event, image_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        unet_pancreas = UNet(1, 1).to(device)
        unet_pancreas.load_state_dict(torch.load(MODEL_PANCREAS_PATH))
        unet_pancreas.eval()

        unet_tumor = UNet(1, 1).to(device)
        unet_tumor.load_state_dict(torch.load(MODEL_UNET_TUMOR_PATH))
        unet_tumor.eval()

        resnet_tumor = ResUNet(1, 1).to(device)
        resnet_tumor.load_state_dict(torch.load(MODEL_RESNET_TUMOR_PATH))
        resnet_tumor.eval()

        attention_tumor = AttentionUNet(1, 1).to(device)
        attention_tumor.load_state_dict(torch.load(MODEL_ATTENTION_TUMOR_PATH))
        attention_tumor.eval()


        predict_images(image_path, unet_pancreas, unet_tumor, resnet_tumor, attention_tumor)

    def display_next_images(self):
        self.current_index += 10
        self.display_images()

    def display_previous_images(self):
        self.current_index -= 10
        self.display_images()


if __name__ == "__main__":
    MODEL_PANCREAS_PATH = 'data/Pancreas_Segmentation/models/model_4_of_5_ep-50_lr-1e-05_bs-16_margin-20.pth'
    MODEL_UNET_TUMOR_PATH = 'data/Pancreas_Tumor_Segmentation/models/model_4_of_5_ep-100_lr-1e-05_bs-2_margin-40.pth'
    MODEL_RESNET_TUMOR_PATH = 'data/Pancreas_Tumor_Segmentation/models/model_resnet_4_of_5_ep-100_lr-1e-05_bs-2_margin-40.pth'
    MODEL_ATTENTION_TUMOR_PATH = 'data/Pancreas_Tumor_Segmentation/models/model_attention_4_of_5_ep-100_lr-1e-05_bs-4_margin-40.pth'
    IMAGES_PATH = "data/Pancreas_Tumor_Segmentation/dataset/images"
    root = tk.Tk()
    app = TumorSegmentationDisplay(root)
    root.mainloop()
