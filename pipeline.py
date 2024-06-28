#inainte de prima rulare, trebuie redenumite 2 fisiere in setul de date: 81 in 25; 82 in 70
#setul descarcat inital are 80 de elemente, 25 si 70 lipsesc.

import sys
import os
import subprocess
from utils import MODELS_PATH

# Parameters
SLICE_TOTAL = 5                     # 4 OR 5;   4 = 75%;    5 = 80%
LOW_RANGE = -100
HIGH_RANGE = 240
MARGIN = 40
Z_MAX = 160
# Y_MAX = 286
# X_MAX = 222
Y_MAX = 256
X_MAX = 192
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
SMOOTH = 1e-4
# SMOOTH = 1e-3
MODEL_PATH = f'{MODELS_PATH}/model_{SLICE_TOTAL-1}_of_{SLICE_TOTAL}_ep-{EPOCHS}_lr-{LEARNING_RATE}_bs-{BATCH_SIZE}_margin-{MARGIN}.pth'
MODEL_PANCREAS_PATH = 'data/Pancreas_Segmentation/models/model_4_of_5_ep-50_lr-1e-05_bs-16_margin-20.pth'
MODEL_TUMOR_PATH = 'data/Pancreas_Tumor_Segmentation/models/model_4_of_5_ep-100_lr-1e-05_bs-2_margin-20.pth'
PREDICT_ONE_IMAGE = "data/Pancreas_Tumor_Segmentation\dataset\images\\0000\\0000.npy"

# Programs
python_cmd = "python"

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
    str(SLICE_TOTAL), str(EPOCHS), str(LEARNING_RATE),
    str(SMOOTH), str(BATCH_SIZE), str(MARGIN)
]

# Test the model
test_cmd = [
    python_cmd, "test.py",
    str(SLICE_TOTAL), str(MODEL_PATH),
    str(SMOOTH), str(MARGIN)
]
predict_cmd = [
    python_cmd, "predict.py",
    str(MODEL_PANCREAS_PATH), str(MODEL_TUMOR_PATH),
    str(MARGIN), str(Y_MAX), str(X_MAX),
]
predict_one_cmd = [
    python_cmd, "predict_one.py",
    str(MODEL_PANCREAS_PATH), str(MODEL_TUMOR_PATH),
    str(MARGIN), str(Y_MAX), str(X_MAX), str(PREDICT_ONE_IMAGE)
]

# Select which command to run based on command line arguments
if len(sys.argv) != 2:
    print("Usage: python pipeline.py [command]")
    sys.exit(1)

command_to_run = sys.argv[1]

# Run selected command
if command_to_run == "convert":
    print("Running data preprocessor...")
    subprocess.run(convert_cmd)
elif command_to_run == "slice":
    print("Running slice.py...")
    subprocess.run(slice_cmd)
elif command_to_run == "data":
    print("Running data.py...")
    subprocess.run(data_cmd)
elif command_to_run == "train":
    print("Running train.py...")
    subprocess.run(train_cmd)
elif command_to_run == "test":
    print("Running test.py...")
    subprocess.run(test_cmd)
elif command_to_run == "predict":
    print("Running predict.py...")
    subprocess.run(predict_cmd)
elif command_to_run == "predict_one":
    print("Running predict_one.py...")
    subprocess.run(predict_one_cmd)
else:
    print("Invalid command:", command_to_run)
    print("Available commands: convert, slice, data, train, test, predict")
