import numpy as np
import os

def display_min_max_values(training_list_path):
    maxim = -100

    with open(training_list_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            file_id, slice_id, image_filename, mask_filename, *_ = line.split()
            if int(file_id) > 0:
                image = np.load(image_filename)
                if -100 < int(image.max()):
                    maxim = max(maxim, float(image.max()))
                    print(f'ID = {file_id}, slice = {slice_id}, MIN = {image.min()}, MAX = {image.max()}')
                # mask = np.load(mask_filename)
                # image_min = min(image_min, image.min())
                # image_max = max(image_max, image.max())
                # mask_min = min(mask_min, mask.min())
                # mask_max = max(mask_max, mask.max())


if __name__ == '__main__':
    training_list_path = 'data/Pancreas_Segmentation/training_Z.txt'
    display_min_max_values(training_list_path)
