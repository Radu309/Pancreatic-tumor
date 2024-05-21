import os
import time

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import load_train_data
from tqdm import tqdm
from utils import model_path


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def up_conv_block(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.encoder1 = conv_block(1, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.bottleneck = conv_block(512, 1024)
        self.decoder4 = conv_block(1024, 512)
        self.decoder3 = conv_block(512, 256)
        self.decoder2 = conv_block(256, 128)
        self.decoder1 = conv_block(128, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv4 = up_conv_block(1024, 512)
        self.upconv3 = up_conv_block(512, 256)
        self.upconv2 = up_conv_block(256, 128)
        self.upconv1 = up_conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))
        e4 = self.encoder4(self.maxpool(e3))

        # Bottleneck
        b = self.bottleneck(self.maxpool(e4))

        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        # Final Convolution
        out = self.final_conv(d1)
        return out


def dice_coef_loss(pred, target):
    smooth = 1.0
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_model(fold, batch_size, epochs, lr):
    dataset = load_train_data(fold)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = UNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = dice_coef_loss
    for epoch in range(epochs):
        model.train()
        total_images = 0
        total_time = 0.0
        epoch_loss = 0
        print(f"\nStarting epoch {epoch + 1}/{epochs}")
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for i, (images, masks) in enumerate(dataloader):
                start_time = time.time()

                images = images.unsqueeze(1).cuda()
                masks = masks.unsqueeze(1).cuda()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                # Accumulate the loss for this epoch
                batch_loss = loss.item()
                epoch_loss += batch_loss
                total_images += images.size(0)

                # Calculate batch processing time
                batch_time = time.time() - start_time
                total_time += batch_time

                # Update progress bar
                avg_time_per_image = total_time / total_images
                pbar.set_postfix(loss=batch_loss, avg_time_per_image=f"{avg_time_per_image:.4f} s")

                pbar.update(1)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}')
    torch.save(model.state_dict(), os.path.join(model_path, f'UNet_fold_{fold}_final.pth'))


if __name__ == "__main__":
    data_path = sys.argv[1]
    folds = int(sys.argv[2])
    epochs = int(sys.argv[3])
    init_lr = float(sys.argv[4])

    print('Using device: ', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    for fold_nr in range(folds):
        train_model(fold_nr, batch_size=1, epochs=epochs, lr=init_lr)
    print("Training done")
