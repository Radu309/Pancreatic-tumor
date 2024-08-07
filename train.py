import logging
import time

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from unet import UNet
from attention import AttentionUNet
from resnet import ResUNet
from utils import *
from data import load_train_and_val_data
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_epoch(model, data_loader, optimizer, device, smooth, is_training=True):
    if is_training:
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    epoch_dice = 0
    epoch_iou = 0
    epoch_precision = 0
    epoch_recall = 0
    epoch_specificity = 0

    for images, masks in tqdm(data_loader, desc="Progress", leave=False):
        if torch.isnan(images).any() or torch.isnan(masks).any():
            raise ValueError("Data contains NaN values.")

        images = images.to(device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device, dtype=torch.float32, non_blocking=True)
        images = images.unsqueeze(1)

        if is_training:
            optimizer.zero_grad()

        outputs = model(images)
        loss = 1 - dice_coefficient(masks, outputs, smooth)

        if is_training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    num_batches = len(data_loader)
    return {
        'loss': epoch_loss / num_batches,
        'dice': epoch_dice / num_batches,
        'iou': epoch_iou / num_batches,
        'precision': epoch_precision / num_batches,
        'recall': epoch_recall / num_batches,
        'specificity': epoch_specificity / num_batches,
        'f1': f1_score(epoch_precision / num_batches, epoch_recall / num_batches)
    }


def train(model_name, slice_total, epochs, learning_rate, smooth, batch_size, margin):
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Smooth:          {smooth}
        Device:          {device}
        Model's path:    {MODELS_PATH}
    ''')

    # Initialize TensorBoard
    writer_log_dir = os.path.join(
        METRICS_PATH,
        f'metrics_{model_name}_{slice_total - 1}_of_{slice_total}_ep-{epochs}_lr-{learning_rate}_bs-{batch_size}_margin-{margin}'
    )
    writer = SummaryWriter(log_dir=f'{writer_log_dir}')

    # Load and preprocess training data with augmentation
    logging.info('\t\tLoading and preprocessing train data...')
    train_dataset, val_dataset = load_train_and_val_data(slice_total, margin)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Create and compile model
    logging.info('\t\tCreating and compiling model...')
    if model_name == 'UNet':
        model = UNet(1, 1).to(device)
    elif model_name == 'AttentionUNet':
        model = AttentionUNet(1, 1).to(device)
    elif model_name == 'ResUNet':
        model = ResUNet(in_channels=1, out_channels=1).to(device)
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        start_time = time.time()  # Record the start time
        logging.info(f'\t\tEpoch {epoch + 1}/{epochs}')

        # Train for one epoch
        train_metrics = run_epoch(model, train_loader, optimizer, device, smooth, is_training=True)
        logging.info(
            f'Training\t\t - Epoch {epoch + 1}/{epochs}, Dice: {train_metrics["dice"]}, IoU: {train_metrics["iou"]}')

        # Log training metrics
        writer.add_scalar(f'Loss/train', train_metrics['loss'], epoch + 1)
        writer.add_scalar(f'Dice/train', train_metrics['dice'], epoch + 1)
        writer.add_scalar(f'IoU/train', train_metrics['iou'], epoch + 1)
        writer.add_scalar(f'Precision/train', train_metrics['precision'], epoch + 1)
        writer.add_scalar(f'Recall/train', train_metrics['recall'], epoch + 1)
        writer.add_scalar(f'Specificity/train', train_metrics['specificity'], epoch + 1)
        writer.add_scalar(f'F1-Score/train', train_metrics['f1'], epoch + 1)

        # Validate for one epoch
        val_metrics = run_epoch(model, validation_loader, optimizer, device, smooth, is_training=False)
        logging.info(
            f'Validation\t\t - Epoch {epoch + 1}/{epochs}, Dice: {val_metrics["dice"]}, IoU: {val_metrics["iou"]}')

        # Log validation metrics
        writer.add_scalar(f'Loss/val', val_metrics['loss'], epoch + 1)
        writer.add_scalar(f'Dice/val', val_metrics['dice'], epoch + 1)
        writer.add_scalar(f'IoU/val', val_metrics['iou'], epoch + 1)
        writer.add_scalar(f'Precision/val', val_metrics['precision'], epoch + 1)
        writer.add_scalar(f'Recall/val', val_metrics['recall'], epoch + 1)
        writer.add_scalar(f'Specificity/val', val_metrics['specificity'], epoch + 1)
        writer.add_scalar(f'F1-Score/val', val_metrics['f1'], epoch + 1)

        # Save the best model
        if (epoch + 1) % 10 == 0:
            model_save_path = os.path.join(
                MODELS_PATH,
                f'model_{model_name}_{slice_total - 1}_of_{slice_total}_ep-{epoch + 1}_lr-{learning_rate}_bs-{batch_size}_margin-{margin}.pth'
            )
            torch.save(model.state_dict(), model_save_path)

        # Update learning rate based on validation loss
        scheduler.step()

        end_time = time.time()
        epoch_time = end_time - start_time
        logging.info(f'\t\tEpoch {epoch + 1} completed in {epoch_time:.2f} seconds')

    writer.close()


if __name__ == "__main__":
    model_name = sys.argv[1]
    slice_total = int(sys.argv[2])
    epochs = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    smooth = float(sys.argv[5])
    batch_size = int(sys.argv[6])
    margin = int(sys.argv[7])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        train(model_name, slice_total, epochs, learning_rate, smooth, batch_size, margin)
        print("Training done")
    else:
        print("Can't start on gpu")
