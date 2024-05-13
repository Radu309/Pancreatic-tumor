import os
import subprocess

# Parameters
cur_fold = 0
FOLDS = 4
LOW_RANGE = -100
HIGH_RANGE = 240
ORGAN_NUMBER = 1
MARGIN = 20
ZMAX = 160
YMAX = 256
XMAX = 192
epoch = 10
init_lr = 1e-5
model_test = f"unet_fd{cur_fold}_Z_ep{epoch}_lr{init_lr}"
vis = False

# Programs
python_cmd = "python"

# Convert masks and images to npy
convert_cmd = [
    python_cmd, "convert_masks_and_images.py"
]

# Slice the 3D volume to 2D slices
slice_cmd = [
    python_cmd, "slice.py",
    str(ORGAN_NUMBER), str(FOLDS), str(LOW_RANGE), str(HIGH_RANGE)
]

# Create data for training
data_cmd = [
    python_cmd, "data.py",
    str(cur_fold), "Z", str(ZMAX), str(YMAX), str(XMAX), str(MARGIN),
    str(ORGAN_NUMBER), str(LOW_RANGE), str(HIGH_RANGE)
]

# Train the model
train_cmd = [
    python_cmd, "train.py",
    str(cur_fold), "Z", str(epoch), str(init_lr)
]

# Test the model
test_cmd = [
    python_cmd, "testvis.py",
    str(model_test), str(cur_fold), "Z",
    str(ZMAX), str(YMAX), str(XMAX), str(HIGH_RANGE), str(LOW_RANGE), str(MARGIN), str(vis)
]

# Run commands
print("Running data preprocessor...")
subprocess.run(convert_cmd)

print("Running slice.py...")
subprocess.run(slice_cmd)

print("Running data.py...")
subprocess.run(data_cmd)

print("Running train.py...")
subprocess.run(train_cmd)

print("Running testvis.py...")
subprocess.run(test_cmd)
