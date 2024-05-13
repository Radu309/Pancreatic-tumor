import os
import subprocess

# Directory paths
DIR = "/Users/yijun/panc"
DATADIR = os.path.join(DIR, "data")
CODEDIR = os.path.join(DIR, "repo")

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

# Slice the 3D volume to 2D slices
slice_cmd = [
    python_cmd, "slice.py",
    str(DATADIR), str(ORGAN_NUMBER), str(FOLDS), str(LOW_RANGE), str(HIGH_RANGE)
]

# Create data for training
data_cmd = [
    python_cmd, "data.py",
    str(DATADIR), str(cur_fold), "Z", str(ZMAX), str(YMAX), str(XMAX), str(MARGIN),
    str(ORGAN_NUMBER), str(LOW_RANGE), str(HIGH_RANGE)
]

# Train the model
train_cmd = [
    python_cmd, "unet.py",
    str(DATADIR), str(cur_fold), "Z", str(epoch), str(init_lr)
]

# Test the model
test_cmd = [
    python_cmd, "testvis.py",
    str(DATADIR), str(model_test), str(cur_fold), "Z",
    str(ZMAX), str(YMAX), str(XMAX), str(HIGH_RANGE), str(LOW_RANGE), str(MARGIN), str(vis)
]

# Run commands
print("Running slice.py...")
subprocess.run(slice_cmd)

print("Running data.py...")
subprocess.run(data_cmd)

print("Running unet.py...")
subprocess.run(train_cmd)

print("Running testvis.py...")
subprocess.run(test_cmd)
