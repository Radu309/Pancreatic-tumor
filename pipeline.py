#inainte de prima rulare, trebuie redenumite 2 fisiere in setul de date: 81 in 25; 82 in 70
#setul descarcat inital are 80 de elemente, 25 si 70 lipsesc.

import sys
import os
import subprocess
from utils import MODELS_PATH

# Parameters
SLICE_TOTAL = 5                     # 4 OR 5
SLICE_FILE = 1                      # HOW TO SLICE THE DATA IN TRAIN AND TEST: X_TRAIN; 100-X_TEST
LOW_RANGE = -100
HIGH_RANGE = 240
MARGIN = 20
Z_MAX = 160
Y_MAX = 400
X_MAX = 400
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
SMOOTH = 1e-3
MODEL_PATH = f'{MODELS_PATH}/training_{SLICE_TOTAL-1}_of_{SLICE_TOTAL}_ep-{EPOCHS}_lr-{LEARNING_RATE}_bs-{BATCH_SIZE}.pth'
# vis = False

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
    str(X_MAX), str(LOW_RANGE), str(HIGH_RANGE)
]

# Train the model
train_cmd = [
    python_cmd, "train.py",
    str(SLICE_TOTAL), str(EPOCHS), str(LEARNING_RATE),
    str(SMOOTH), str(BATCH_SIZE)
]

# Test the model
test_cmd = [
    python_cmd, "test.py",
    str(SLICE_FILE), str(SLICE_TOTAL), str(MODEL_PATH),
    str(SMOOTH)
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
else:
    print("Invalid command:", command_to_run)
    print("Available commands: convert, slice, data, train, test")
