import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    # def _crop(self, enc, dec):
    #     _, _, H, W = dec.size()
    #     enc = F.interpolate(enc, size=(H, W), mode='bilinear', align_corners=False)
    #     return enc
    #
    #
    # def forward(self, x):
    #     enc1 = self.encoder1(x)
    #     enc2 = self.encoder2(self.pool1(enc1))
    #     enc3 = self.encoder3(self.pool2(enc2))
    #     enc4 = self.encoder4(self.pool3(enc3))
    #     bottleneck = self.bottleneck(self.pool4(enc4))
    #
    #     dec4 = self.upconv4(bottleneck)
    #     dec4 = torch.cat((dec4, self._crop(enc4, dec4)), dim=1)
    #     dec4 = self.decoder4(dec4)
    #     dec3 = self.upconv3(dec4)
    #     dec3 = torch.cat((dec3, self._crop(enc3, dec3)), dim=1)
    #     dec3 = self.decoder3(dec3)
    #     dec2 = self.upconv2(dec3)
    #     dec2 = torch.cat((dec2, self._crop(enc2, dec2)), dim=1)
    #     dec2 = self.decoder2(dec2)
    #     dec1 = self.upconv1(dec2)
    #     dec1 = torch.cat((dec1, self._crop(enc1, dec1)), dim=1)
    #     dec1 = self.decoder1(dec1)
    #     return self.sigmoid(self.final_conv(dec1))

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
        return self.sigmoid(self.final_conv(dec1))
