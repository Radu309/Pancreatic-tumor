import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import *
from data import load_train_and_val_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Smooth:          {smooth}
        Device:          {device}
        Model's path:    {current_models_path}
    ''')
    # Initialize TensorBoard
    writer_log_dir = os.path.join(current_logs_path, f'fold-{fold}_ep-{epochs}_lr-{learning_rate}_bs-{batch_size}')
    writer = SummaryWriter(log_dir=f'{writer_log_dir}')

    # --------------------- load and preprocess training data -----------------
    logging.info('\t\tLoading and preprocessing train data...')

    train_dataset, val_dataset = load_train_and_val_data(fold)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ---------------------- Create, compile, and train model ------------------------
    logging.info('\t\tCreating and compiling model...')
    model = UNet(1, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # ---------------------- Fitting model ------------------------
    logging.info('\t\tFitting model...')
    metrics = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        epoch_iou = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for images, masks in train_loader:
                if torch.isnan(images).any() or torch.isnan(masks).any():
                    raise ValueError("Data contains NaN values.")
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                images = images.unsqueeze(1)  # Add channel dimension: [batch_size, 1, height, width]
                optimizer.zero_grad()
                outputs = model(images)
                loss = 1 - dice_coefficient(masks, outputs, smooth)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                epoch_loss += loss.item()

                # Calculate metrics
                dice = dice_coefficient(masks, outputs, smooth).item()
                iou = iou_score(masks, outputs)
                epoch_dice += dice
                epoch_iou += iou

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)
        epoch_loss /= len(train_loader)
        epoch_dice /= len(train_loader)
        epoch_iou /= len(train_loader)
        metrics.append([epoch + 1, epoch_loss, epoch_dice, epoch_iou])

        logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, Dice: {epoch_dice}, IoU: {epoch_iou}')
        # Log metrics to Weights & Biases
        writer.add_scalar('Loss/train', epoch_loss, epoch + 1)
        writer.add_scalar('Dice/train', epoch_dice, epoch + 1)
        writer.add_scalar('IoU/train', epoch_iou, epoch + 1)

        # Perform validation
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        with torch.no_grad():
            for images, masks in validation_loader:
                images, masks = images.cuda(), masks.cuda()
                images = images.unsqueeze(1)  # Add channel dimension: [batch_size, 1, height, width]

                outputs = model(images)
                loss = 1 - dice_coefficient(masks, outputs, smooth)
                val_loss += loss.item()
                dice = dice_coefficient(masks, outputs, smooth).item()
                iou = iou_score(masks, outputs)
                val_dice += dice
                val_iou += iou
        val_loss /= len(validation_loader)
        val_dice /= len(validation_loader)
        val_iou /= len(validation_loader)

        logging.info(f'Validation - Epoch {epoch + 1}/{epochs}, Loss: {val_loss}, Dice: {val_dice}, IoU: {val_iou}')
        writer.add_scalar('Loss/val', val_loss, epoch + 1)
        writer.add_scalar('Dice/val', val_dice, epoch + 1)
        writer.add_scalar('IoU/val', val_iou, epoch + 1)

        # Save the model at regular intervals
        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(current_models_path,
                                           f'fold-{fold}_ep-{epoch + 1}_lr-{learning_rate}_bs-{batch_size}.pth')
            torch.save(model.state_dict(), model_save_path)
    writer.close()


if __name__ == "__main__":
    data_path = sys.argv[1]
    folds = int(sys.argv[2])
    epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    smooth = float(sys.argv[5])
    batch_size = int(sys.argv[6])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        next_execution_id = get_next_execution_id()
        current_models_path = os.path.join(execution_path, f'execution_{next_execution_id}', 'models')
        current_logs_path = os.path.join(execution_path, f'execution_{next_execution_id}', 'runs')
        os.makedirs(current_models_path, exist_ok=True)
        os.makedirs(current_logs_path, exist_ok=True)
        for fold_nr in range(folds):
            train(fold_nr)
        print("Training done")
    else:
        print("Can't start on gpu")
