import numpy as np
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *
from data import load_train_data


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.final_conv(dec1)


def train(fold):
    print(f"Number of epoch: {epochs}")
    print(f"Learning rate: {init_lr}")

    # --------------------- load and preprocess training data -----------------
    print('\n        Loading and preprocessing train data...')
    dataloader = load_train_data(fold)

    # ---------------------- Create, compile, and train model ------------------------
    print('        Creating and compiling model...')
    model = UNet(1, 1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)

    print('        Fitting model...\n')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for images, masks in dataloader:
                images, masks = images.cuda(), masks.cuda()
                # Ensure input has the shape [batch_size, channels, height, width]
                if len(images.shape) == 3:  # Case when input is [height, width, channels]
                    images = images.unsqueeze(1)  # Add channel dimension: [batch_size, 1, height, width]
                elif len(images.shape) == 2:  # Case when input is [height, width]
                    images = images.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions: [1, 1, height, width]
                optimizer.zero_grad()
                outputs = model(images)
                loss = -dice_coefficient(masks, outputs, smooth)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}')

        # Save the model at regular intervals
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_path, f'unet_fold_{fold}_Z_ep{epoch + 1}_lr{init_lr}.pth'))


if __name__ == "__main__":
    data_path = sys.argv[1]
    folds = int(sys.argv[2])
    epochs = int(sys.argv[3])
    init_lr = float(sys.argv[4])
    smooth = float(sys.argv[5])
    batch_size = int(sys.argv[6])

    print('Using device: ', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if torch.cuda.is_available():
        for fold_nr in range(folds):
            train(fold_nr)
        print("Training done")
    else:
        print("Can't start on cpu")
