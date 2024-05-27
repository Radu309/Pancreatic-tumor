#inainte de prima rulare, trebuie redenumite 2 fisiere in setul de date: 81 in 25; 82 in 70
#setul descarcat inital are 80 de elemente, 25 si 70 lipsesc.

import sys
import os
import subprocess


data_path = "data/Pancreas_Segmentation"
MODEL_PATH = "data/Pancreas_Segmentation/models/fold-0_ep-50_lr-1e-05_bs-16.pth"

# Define paths at the module level
dataset_path = os.path.join(data_path, 'dataset')
image_path = os.path.join(dataset_path, 'images')
mask_path = os.path.join(dataset_path, 'masks')
image_npy_path = os.path.join(dataset_path, 'NPY_Images')
mask_npy_path = os.path.join(dataset_path, 'NPY_Masks')

dataloader_path = os.path.join(data_path, 'dataloader')
train_dataloader_path = os.path.join(dataloader_path, 'train')
test_dataloader_path = os.path.join(dataloader_path, 'test')

lists_path = os.path.join(data_path, 'lists')
model_path = os.path.join(data_path, 'executions')
predicted_path = os.path.join(data_path, 'predicted')

# Ensure directories exist
paths = [dataset_path, image_path, mask_path, image_npy_path, mask_npy_path, dataloader_path,
         train_dataloader_path, test_dataloader_path, lists_path, model_path, predicted_path]
for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)

list_dataset = os.path.join(lists_path, 'dataset' + '.txt')

# Parameters
CURRENT_FOLDER = 0
FOLDS = 4
LOW_RANGE = -100
HIGH_RANGE = 240
MARGIN = 20
Z_MAX = 160
Y_MAX = 256
X_MAX = 192
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
SMOOTH = 1e-3
# model_test = f"unet_fd{CURRENT_FOLDER}_Z_ep{EPOCHS}_lr{LEARNING_RATE}"
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
    str(FOLDS), str(LOW_RANGE), str(HIGH_RANGE), str(image_npy_path), str(image_path),
    str(mask_npy_path), str(mask_path), str(list_dataset), str(lists_path)
]

# Create data for training
data_cmd = [
    python_cmd, "data.py",
    str(FOLDS), str(Z_MAX), str(Y_MAX), str(X_MAX), str(MARGIN),
    str(LOW_RANGE), str(HIGH_RANGE)
]

# Train the model
train_cmd = [
    python_cmd, "train.py",
    str(FOLDS),  str(EPOCHS), str(LEARNING_RATE), str(SMOOTH), str(BATCH_SIZE),
]

# Test the model
test_cmd = [
    python_cmd, "test.py",
    str(CURRENT_FOLDER), str(MODEL_PATH), str(SMOOTH)
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
