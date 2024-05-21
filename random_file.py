import os

import numpy as np
import torch
from torch.utils.data import DataLoader


# from data import PrepareDataset


def display_dataloader(dataloader):
    print(len(dataloader))
    count = 0
    for i, (images, masks) in enumerate(dataloader):
        if images.max().item() > -100:
            print("Images ID: ", i, " value = ", images.max().item())
            count = count + 1
        # if masks.max().item() == 1.0:
        #     print("Masks ID: ", i)
    print(count)


if __name__ == '__main__':
    data_path = 'data/Pancreas_Segmentation/train/'
    # fold = int(sys.argv[2])
    # plane = sys.argv[3]

    # dataset = torch.load(os.path.join(data_path, f'dataset/train_dataset_fold_1_plane_Z.pt'))
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    #
    # display_dataloader(dataloader)
    print(np.zeros((10), dtype=[('x', 'int'), ('y', 'float')]))
