import numpy as np
import os


def display_min_max_values(training_list_path, z_start, z_end):
    maxim = -100

    image = np.load(training_list_path)

    for i in range(z_start, z_end):
        image_slice = image[:, :, i]
        if maxim < int(image_slice.max()):
            maxim = max(maxim, float(image.max()))
            print(f'Slice = {i}, MIN = {image_slice.min()}, MAX = {image_slice.max()}')


if __name__ == '__main__':
    training_list_path = 'data/Pancreas_Segmentation/NPY_Images/0001.npy'
    z_start = 63
    z_end = 131
    display_min_max_values(training_list_path, z_start, z_end)
