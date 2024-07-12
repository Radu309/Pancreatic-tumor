#inainte de prima rulare, trebuie redenumite 2 fisiere in setul de date: 81 in 25; 82 in 70
#setul descarcat inital are 80 de elemente, 25 si 70 lipsesc.

import sys
import os
import subprocess
from utils import MODELS_PATH

# Parameters
SLICE_TOTAL = 5
LOW_RANGE = -100
HIGH_RANGE = 240
MARGIN = 40
Z_MAX = 160
Y_MAX = 256
X_MAX = 192
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
SMOOTH = 1e-4
MODEL_NAMES = ['UNet', 'ResUNet', 'AttentionUNet']  #unet = 0, resunet = 1, attention = 2.
NUMBER_MODEL = 2
MODEL_TYPE = MODEL_NAMES[NUMBER_MODEL]
MODEL_PATH = f'{MODELS_PATH}/model_{MODEL_TYPE}_{SLICE_TOTAL-1}_of_{SLICE_TOTAL}_ep-{EPOCHS}_lr-{LEARNING_RATE}_bs-{BATCH_SIZE}_margin-{MARGIN}.pth'
#for app and predict_all
MODEL_TUMOR_PATH = 'data/Pancreas_Tumor_Segmentation/models/model_UNet_4_of_5_ep-100_lr-1e-05_bs-2_margin-40.pth'
MODEL_PANCREAS_PATH = 'data/Pancreas_Segmentation/models/model_UNet_4_of_5_ep-50_lr-1e-05_bs-16_margin-20.pth'
MODEL_UNET_PANCREAS_PATH = 'data/Pancreas_Segmentation/models/model_UNet_4_of_5_ep-50_lr-1e-05_bs-16_margin-20.pth'
MODEL_UNET_TUMOR_PATH = 'data/Pancreas_Tumor_Segmentation/models/model_UNet_4_of_5_ep-100_lr-1e-05_bs-2_margin-40.pth'
MODEL_RESNET_TUMOR_PATH = 'data/Pancreas_Tumor_Segmentation/models/model_ResUNet_4_of_5_ep-100_lr-1e-05_bs-2_margin-40.pth'
MODEL_ATTENTION_TUMOR_PATH = 'data/Pancreas_Tumor_Segmentation/models/model_AttentionUNet_4_of_5_ep-100_lr-1e-05_bs-2_margin-40.pth'
MODEL_PREDICTS = [MODEL_UNET_TUMOR_PATH, MODEL_RESNET_TUMOR_PATH, MODEL_ATTENTION_TUMOR_PATH]
MODEL_PREDICT = MODEL_PREDICTS[NUMBER_MODEL]
IMAGES_PATH = "data/Pancreas_Tumor_Segmentation/dataset/images"
#

# Programs
python_cmd = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")

# Convert masks and images to npy
convert_cmd = [
    python_cmd, "convert_to_numpy.py"
]

# Slice the 3D volume to 2D slices
slice_cmd = [
    python_cmd, "slice.py",
    str(LOW_RANGE), str(HIGH_RANGE)
]

# Create data for training
data_cmd = [
    python_cmd, "data.py",
    str(SLICE_TOTAL), str(Z_MAX), str(Y_MAX),
    str(X_MAX), str(MARGIN)
]

# Train the model
train_cmd = [
    python_cmd, "train.py",
    str(MODEL_TYPE), str(SLICE_TOTAL), str(EPOCHS), str(LEARNING_RATE),
    str(SMOOTH), str(BATCH_SIZE), str(MARGIN)
]

# Test the model
test_cmd = [
    python_cmd, "test.py",
    str(MODEL_TYPE), str(SLICE_TOTAL), str(MODEL_PATH),
    str(SMOOTH), str(MARGIN)
]
predict_cmd = [
    python_cmd, "predict.py",
    str(MODEL_TYPE), str(MODEL_PANCREAS_PATH), str(MODEL_PREDICT),
    str(MARGIN), str(Y_MAX), str(X_MAX),
]
app_cmd = [
    python_cmd, "app.py",
    str(MODEL_UNET_PANCREAS_PATH), str(MODEL_UNET_TUMOR_PATH), str(MODEL_RESNET_TUMOR_PATH),
    str(MODEL_ATTENTION_TUMOR_PATH), str(IMAGES_PATH), str(MARGIN), str(Y_MAX), str(X_MAX),
]
# Select which command to run based on command line arguments
if len(sys.argv) != 2:
    print("Usage: python pipeline.py [command]")
    sys.exit(1)

command_to_run = sys.argv[1]

# Run selected command
if command_to_run == "convert":
    subprocess.run(convert_cmd)
elif command_to_run == "slice":
    subprocess.run(slice_cmd)
elif command_to_run == "data":
    subprocess.run(data_cmd)
elif command_to_run == "train":
    subprocess.run(train_cmd)
elif command_to_run == "test":
    subprocess.run(test_cmd)
elif command_to_run == "predict":
    subprocess.run(predict_cmd)
elif command_to_run == "app":
    subprocess.run(app_cmd)
else:
    print("Invalid command:", command_to_run)
    print("Available commands: convert, slice, data, train, test, predict, app")
