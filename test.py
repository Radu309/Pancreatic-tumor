import sys
import os
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet import UNet
from utils import dice_coefficient, iou_score, precision_score, PREDICTED_PATH
from data import load_test_data
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_image(image, mask, output, save_dir, idx):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Input Image')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Ground Truth Mask')

    ax[2].imshow(output, cmap='gray')
    ax[2].set_title('Predicted Mask')

    plt.savefig(os.path.join(save_dir, f'image_{idx}.png'))
    plt.close(fig)


def test():
    logging.info(f'\t\tStarting testing for slice file = {slice_file}_of_{slice_total}')

    # Load test data
    logging.info('\t\tLoading and preprocessing test data...')
    test_dataset = load_test_data(slice_file, slice_total)
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

    # Metrics
    test_dice = 0
    test_iou = 0
    test_precision = 0

    # Testing loop
    logging.info('\t\tEvaluating the model...')
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
            for idx, (images, masks) in enumerate(test_loader):
                images, masks = images.to(device), masks.to(device)
                images = images.unsqueeze(1)
                outputs = model(images)

                save_image(images.cpu().numpy().squeeze(), masks.cpu().numpy().squeeze(),
                           outputs.cpu().numpy().squeeze(), predicted_model_dir, idx)

                # Calculate metrics
                dice = dice_coefficient(masks, outputs, smooth).item()
                iou = iou_score(masks, outputs)
                precision = precision_score(masks.cpu().numpy().squeeze(), outputs.cpu().numpy().squeeze())

                test_dice += dice
                test_iou += iou
                test_precision += precision

                pbar.update(1)

    test_dice /= len(test_loader)
    test_iou /= len(test_loader)
    test_precision /= len(test_loader)
    logging.info(f'Testing completed - percent {slice_file}_of_{slice_total}, Precision: {test_precision}')


if __name__ == "__main__":
    slice_file = int(sys.argv[1])
    slice_total = int(sys.argv[2])
    model_path = sys.argv[3]

    smooth = float(sys.argv[5])
    low_range = int(sys.argv[6])
    high_range = int(sys.argv[7])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        test()
        print("Testing done")
    else:
        print("Can't start on gpu")

