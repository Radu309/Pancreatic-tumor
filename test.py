import csv
import sys
import os
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from utils import dice_coefficient, iou_score, precision_score, PREDICTED_PATH, METRICS_PATH
from data import load_test_data
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_image(image, mask, output, save_dir, idx):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    im0 = ax[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title('Input Image')
    fig.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title('Ground Truth Mask')
    fig.colorbar(im1, ax=ax[1])

    im2 = ax[2].imshow(output, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title('Predicted Mask')
    fig.colorbar(im2, ax=ax[2])

    plt.savefig(os.path.join(save_dir, f'image_{idx}.png'))
    plt.close(fig)


def test():
    logging.info(f'\t\tStarting testing for slice file = 1_of_{slice_total}')

    # Load test data
    logging.info('\t\tLoading and preprocessing test data...')
    test_dataset = load_test_data(slice_total, margin)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load the model
    logging.info('\t\tLoading the model...')
    model = UNet(1, 1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create directory for saving predictions
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    predicted_model_dir = os.path.join(PREDICTED_PATH, model_name)
    os.makedirs(predicted_model_dir, exist_ok=True)

    results_file_path = os.path.join(METRICS_PATH, f'{model_name}.csv')
    logging.info('\t\tEvaluating the model...')
    with open(results_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')  # Set the delimiter to semicolon
        writer.writerow(['Batch Index', 'Dice Coefficient', 'IOU Score', 'Precision', 'Accuracy'])

        total_dice = 0
        total_iou = 0
        total_precision = 0
        total_accuracy = 0
        with torch.no_grad(), tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
            for idx, (images, masks) in enumerate(test_loader):
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                masks = masks.to(device, dtype=torch.float32, non_blocking=True)
                images = images.unsqueeze(1)
                outputs = model(images)
                outputs_prob = torch.sigmoid(outputs)

                save_image(images.cpu().numpy().squeeze(), masks.cpu().numpy().squeeze(),
                           outputs.cpu().numpy().squeeze(), predicted_model_dir, idx)

                dice = dice_coefficient(masks, outputs, smooth).item()
                iou = iou_score(masks, outputs)
                precision = precision_score(masks, outputs).item()
                accuracy = (outputs_prob.round() == masks).float().mean().item()
                writer.writerow([idx, dice, iou, precision, accuracy])

                total_dice += dice
                total_iou += iou
                total_precision += precision
                total_accuracy += accuracy

                pbar.update(1)

        # Calculează mediile
        mean_dice = total_dice / len(test_loader)
        mean_iou = total_iou / len(test_loader)
        mean_precision = total_precision / len(test_loader)
        mean_accuracy = total_accuracy / len(test_loader)
        # Scrie mediile în CSV
        writer.writerow(['Average', mean_dice, mean_iou, mean_precision, mean_accuracy])

    logging.info(f'Testing completed - percent 1_of_{slice_total}')


if __name__ == "__main__":
    slice_total = int(sys.argv[1])
    model_path = sys.argv[2]
    smooth = float(sys.argv[3])
    margin = sys.argv[4]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        test()
        print("Testing done")
    else:
        print("Can't start on gpu")

