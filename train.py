import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet import UNet
from utils import *
from data import PrepareDataset, load_train_and_val_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train():
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Smooth:          {smooth}
        Device:          {device}
        Model's path:    {MODELS_PATH}
    ''')
    # Initialize TensorBoard
    writer_log_dir = os.path.join(METRICS_PATH,
                                  f'metrics_{slice_total-1}_of_{slice_total}_ep-{epochs}_lr-{learning_rate}_bs-{batch_size}')
    writer = SummaryWriter(log_dir=f'{writer_log_dir}')

    # --------------------- load and preprocess training data -----------------
    logging.info('\t\tLoading and preprocessing train data...')
    # train_image_filenames = np.load(os.path.join(DATALOADER_PATH, f'training_{slice_total-1}_of_{slice_total}_images.npy'))
    # train_mask_filenames = np.load(os.path.join(DATALOADER_PATH, f'training_{slice_total-1}_of_{slice_total}_masks.npy'))
    # val_image_filenames = np.load(os.path.join(DATALOADER_PATH, f'validation_{slice_total-1}_of_{slice_total}_images.npy'))
    # val_mask_filenames = np.load(os.path.join(DATALOADER_PATH, f'validation_{slice_total-1}_of_{slice_total}_masks.npy'))
    #
    # train_dataset = PrepareDataset(train_image_filenames, train_mask_filenames)
    # val_dataset = PrepareDataset(val_image_filenames, val_mask_filenames)
    #
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    train_dataset, val_dataset = load_train_and_val_data(slice_total)
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
        epoch_precision = 0
        epoch_recall = 0
        epoch_specificity = 0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for images, masks in train_loader:
                if torch.isnan(images).any() or torch.isnan(masks).any():
                    raise ValueError("Data contains NaN values.")
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                masks = masks.to(device, dtype=torch.float32, non_blocking=True)
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
                precision = precision_score(masks, outputs).item()
                recall = recall_score(masks, outputs).item()
                specificity = specificity_score(masks, outputs).item()

                epoch_dice += dice
                epoch_iou += iou
                epoch_precision += precision
                epoch_recall += recall
                epoch_specificity += specificity

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)

        torch.cuda.empty_cache()

        epoch_loss /= len(train_loader)
        epoch_dice /= len(train_loader)
        epoch_iou /= len(train_loader)
        epoch_precision /= len(train_loader)
        epoch_recall /= len(train_loader)
        epoch_specificity /= len(train_loader)
        f1 = f1_score(epoch_precision, epoch_recall)
        metrics.append([epoch + 1, epoch_loss, epoch_dice, epoch_iou, epoch_precision, epoch_recall, epoch_specificity, f1])

        logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, Dice: {epoch_dice}, IoU: {epoch_iou}, Precision: {epoch_precision}, Recall: {epoch_recall}, Specificity: {epoch_specificity}, F1-Score: {f1}')
        # Log metrics to Weights & Biases
        writer.add_scalar('Loss/train', epoch_loss, epoch + 1)
        writer.add_scalar('Dice/train', epoch_dice, epoch + 1)
        writer.add_scalar('IoU/train', epoch_iou, epoch + 1)
        writer.add_scalar('Precision/train', epoch_precision, epoch + 1)
        writer.add_scalar('Recall/train', epoch_recall, epoch + 1)
        writer.add_scalar('Specificity/train', epoch_specificity, epoch + 1)
        writer.add_scalar('F1-Score/train', f1, epoch + 1)

        # Perform validation
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        val_precision = 0
        val_recall = 0
        val_specificity = 0
        with torch.no_grad():
            for images, masks in validation_loader:
                images, masks = images.cuda(), masks.cuda()
                images = images.unsqueeze(1)
                outputs = model(images)
                loss = 1 - dice_coefficient(masks, outputs, smooth)
                val_loss += loss.item()
                dice = dice_coefficient(masks, outputs, smooth).item()
                iou = iou_score(masks, outputs)
                precision = precision_score(masks, outputs).item()
                recall = recall_score(masks, outputs).item()
                specificity = specificity_score(masks, outputs).item()

                val_dice += dice
                val_iou += iou
                val_precision += precision
                val_recall += recall
                val_specificity += specificity

        val_loss /= len(validation_loader)
        val_dice /= len(validation_loader)
        val_iou /= len(validation_loader)
        val_precision /= len(validation_loader)
        val_recall /= len(validation_loader)
        val_specificity /= len(validation_loader)
        val_f1 = f1_score(val_precision, val_recall)

        logging.info(f'Validation - Epoch {epoch + 1}/{epochs}, Loss: {val_loss}, Dice: {val_dice}, IoU: {val_iou}, Precision: {val_precision}, Recall: {val_recall}, Specificity: {val_specificity}, F1-Score: {val_f1}')
        writer.add_scalar('Loss/val', val_loss, epoch + 1)
        writer.add_scalar('Dice/val', val_dice, epoch + 1)
        writer.add_scalar('IoU/val', val_iou, epoch + 1)
        writer.add_scalar('Precision/val', val_precision, epoch + 1)
        writer.add_scalar('Recall/val', val_recall, epoch + 1)
        writer.add_scalar('Specificity/val', val_specificity, epoch + 1)
        writer.add_scalar('F1-Score/val', val_f1, epoch + 1)

        # Save the model at regular intervals
        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(
                MODELS_PATH,
                f'model_{slice_total-1}_of_{slice_total}_ep-{epoch + 1}_lr-{learning_rate}_bs-{batch_size}.pth'
            )
            torch.save(model.state_dict(), model_save_path)
    writer.close()


if __name__ == "__main__":

    slice_total = int(sys.argv[1])
    epochs = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    smooth = float(sys.argv[4])
    batch_size = int(sys.argv[5])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        train()
        print("Training done")
    else:
        print("Can't start on gpu")
