import numpy as np
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

from utils import *
from data import load_train_data

# ----- paths setting -----
data_path = "data/Pancreas_Segmentation/train"
model_path = data_path + "models/"
log_path = data_path + "logs/"

# ----- params for training and testing -----
batch_size = 1
cur_fold = int(sys.argv[2])
plane = sys.argv[3]
epoch = int(sys.argv[4])
init_lr = float(sys.argv[5])

# ----- Dice Coefficient and cost function for training -----
smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_true, y_pred):
        return -dice_coef(y_true, y_pred)


def train(fold, plane, batch_size, nb_epoch, init_lr):
    """
    Train an Unet model with data from load_train_data()

    Parameters
    ----------
    fold : int
        which fold is experimenting in 4-fold. It should be one of 0/1/2/3

    plane : char
        which plane is experimenting. It is from 'X'/'Y'/'Z'

    batch_size : int
        size of mini-batch

    nb_epoch : int
        number of epochs to train NN

    init_lr : float
        initial learning rate
    """
    print("Number of epochs:", nb_epoch)
    print("Learning rate:", init_lr)

    # --------------------- Load and preprocess training data -----------------
    print('-' * 80)
    print('Loading and preprocessing train data...')
    print('-' * 80)

    imgs_train, imgs_mask_train = load_train_data(fold, plane)

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = torch.from_numpy(imgs_train).float()
    imgs_mask_train = torch.from_numpy(imgs_mask_train).float()

    # ---------------------- Create and compile model ------------------------
    print('-' * 80)
    print('Creating and compiling model...')
    print('-' * 80)

    model = UNet()  # Create your UNet model
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    print(model)

    print('-' * 80)
    print('Fitting model...')
    print('-' * 80)

    ver = f'unet_fd{cur_fold}_{plane}_ep{epoch}_lr{init_lr}.csv'
    csv_logger = CSVLogger(log_path + ver)

    for epoch in range(nb_epoch):
        model.train()
        running_loss = 0.0

        for i in range(0, len(imgs_train), batch_size):
            inputs = Variable(imgs_train[i:i + batch_size])
            labels = Variable(imgs_mask_train[i:i + batch_size])

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{nb_epoch}], Loss: {running_loss / len(imgs_train)}')

    print('Training done')


if __name__ == "__main__":
    train(cur_fold, plane, batch_size, epoch, init_lr)
