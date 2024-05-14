import numpy as np
import sys
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import load_train_data
from utils import preprocess


# Define the U-Net architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define the encoder part
        self.enc_conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.enc_conv4 = nn.Conv2d(256, 512, 3, padding=1)
        # Define the decoder part
        self.dec_conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec_conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_conv4 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        enc1 = nn.ReLU()(self.enc_conv1(x))
        enc2 = nn.ReLU()(self.enc_conv2(enc1))
        enc3 = nn.ReLU()(self.enc_conv3(enc2))
        enc4 = nn.ReLU()(self.enc_conv4(enc3))
        # Decoder
        dec1 = nn.ReLU()(self.dec_conv1(enc4))
        dec2 = nn.ReLU()(self.dec_conv2(dec1))
        dec3 = nn.ReLU()(self.dec_conv3(dec2))
        dec4 = nn.Sigmoid()(self.dec_conv4(dec3))
        return dec4


# Define the Dice coefficient loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        return -(2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)


# Define training function
def train(fold, plane, batch_size, nb_epoch, init_lr):
    # Load and preprocess training data
    imgs_train, imgs_mask_train = load_train_data(fold, plane)
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    # Convert data to PyTorch tensors
    imgs_train = torch.from_numpy(imgs_train).unsqueeze(1)
    imgs_mask_train = torch.from_numpy(imgs_mask_train).unsqueeze(1)

    # Initialize U-Net model, Dice loss, and Adam optimizer
    model = UNet()
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    # Define data loader
    train_dataset = torch.utils.data.TensorDataset(imgs_train, imgs_mask_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(nb_epoch):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, masks = data
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, masks.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print statistics
        print('[Epoch %d] Loss: %.3f' %
              (epoch + 1, running_loss / len(train_loader)))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train 2D U-Net')
    parser.add_argument('data_path', type=str, help='Path to data directory')
    parser.add_argument('fold', type=int, help='Fold number (0, 1, 2, or 3)')
    parser.add_argument('plane', type=str, help='Plane (X, Y, or Z)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (default: 1)')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--init_lr', type=float, default=1e-5, help='Initial learning rate (default: 1e-5)')
    args = parser.parse_args()

    # Set data paths
    data_path = args.data_path
    model_path = os.path.join(data_path, "models/")
    log_path = os.path.join(data_path, "logs/")

    # Create directories if not exist
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Train the model
    train(args.fold, args.plane, args.batch_size, args.epoch, args.init_lr)
    print("Training done.")