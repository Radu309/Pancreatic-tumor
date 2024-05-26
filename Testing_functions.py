import torch
import sys
import matplotlib.pyplot as plt
import numpy as np


def display_pt_data():
    # Încărcați datele din fișierul .pt
    # train_dataset, val_dataset = load_train_and_val_data(0)
    images, masks, _, _ = torch.load(file_path)
    # images, masks = train_dataset
    print(len(images))
    print(len(masks))

    # Afișați informații despre dimensiuni
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")

    # Afișați primele imagini și măști
    for i in range(30, 50):
        image = images[i].numpy()
        mask = masks[i].numpy()

        # Afișare imagine
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Image {i + 1}')

        # Afișare mască
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask {i + 1}')

        plt.show()


if __name__ == '__main__':
    file_path = 'data/Pancreas_Segmentation/train/dataset/train_dataset_fold_0_plane_Z.pt'
    display_pt_data()
