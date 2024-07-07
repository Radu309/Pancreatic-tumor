import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.bottleneck = self.conv_block(512, 512)

        # Decoder
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 512)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1_p, indices1 = self.pool1(enc1)
        enc2 = self.enc2(enc1_p)
        enc2_p, indices2 = self.pool2(enc2)
        enc3 = self.enc3(enc2_p)
        enc3_p, indices3 = self.pool3(enc3)
        enc4 = self.enc4(enc3_p)
        enc4_p, indices4 = self.pool4(enc4)
        bottleneck = self.bottleneck(enc4_p)

        # Debug prints
        print(f"enc1 shape: {enc1.shape}, enc1_p shape: {enc1_p.shape}, indices1 shape: {indices1.shape}")
        print(f"enc2 shape: {enc2.shape}, enc2_p shape: {enc2_p.shape}, indices2 shape: {indices2.shape}")
        print(f"enc3 shape: {enc3.shape}, enc3_p shape: {enc3_p.shape}, indices3 shape: {indices3.shape}")
        print(f"enc4 shape: {enc4.shape}, enc4_p shape: {enc4_p.shape}, indices4 shape: {indices4.shape}")
        print(f"bottleneck shape: {bottleneck.shape}")

        # Decoder
        dec4 = self.unpool4(bottleneck, indices4, output_size=enc4.size())
        dec4 = self.dec4(dec4)
        dec3 = self.unpool3(dec4, indices3, output_size=enc3.size())
        dec3 = self.dec3(dec3)
        dec2 = self.unpool2(dec3, indices2, output_size=enc2.size())
        dec2 = self.dec2(dec2)
        dec1 = self.unpool1(dec2, indices1, output_size=enc1.size())
        dec1 = self.dec1(dec1)
        output = self.sigmoid(self.final_conv(dec1))

        # Debug prints for decoder
        print(f"dec4 shape: {dec4.shape}")
        print(f"dec3 shape: {dec3.shape}")
        print(f"dec2 shape: {dec2.shape}")
        print(f"dec1 shape: {dec1.shape}")
        print(f"output shape: {output.shape}")

        return output

# Sample usage and testing
if __name__ == "__main__":
    model = SegNet(in_channels=1, out_channels=1)
    print(model)

    # Sample input
    x = torch.randn(2, 1, 192, 256)
    output = model(x)
    print(output.shape)  # Expected output shape should match input resolution
