import sys
import subprocess

#inainte de prima rulare, trebuie redenumite 2 fisiere in setul de date: 81 in 25; 82 in 70
#setul descarcat are 80 de elemente, 25 si 70 lipsesc.

#online Z plane for dataset

DATADIR_TRAIN = "data/Pancreas_Segmentation/train"

# Parameters
cur_fold = 0
FOLDS = 4
LOW_RANGE = -100
HIGH_RANGE = 240
MARGIN = 20
ZMAX = 160
YMAX = 256
XMAX = 192
epoch = 100
batch_size = 16
init_lr = 1e-8
smooth = 1e-3
model_test = f"unet_fd{cur_fold}_Z_ep{epoch}_lr{init_lr}"
vis = False

# Programs
python_cmd = "python"

# Convert masks and images to npy
convert_cmd = [
    python_cmd, "convert_to_numpy.py"
]

# Slice the 3D volume to 2D slices
slice_cmd = [
    python_cmd, "slice.py",
    str(DATADIR_TRAIN), str(FOLDS), str(LOW_RANGE), str(HIGH_RANGE)
]

# Create data for training
data_cmd = [
    python_cmd, "data.py",
    str(DATADIR_TRAIN), str(FOLDS), str(ZMAX), str(YMAX), str(XMAX), str(MARGIN),
    str(LOW_RANGE), str(HIGH_RANGE)
]

# Train the model
train_cmd = [
    python_cmd, "train.py",
    str(DATADIR_TRAIN), str(FOLDS),  str(epoch), str(init_lr), str(smooth), str(batch_size),
]

# Test the model
test_cmd = [
    python_cmd, "test.py",
    str(model_test), str(cur_fold), "Z",
    str(ZMAX), str(YMAX), str(XMAX), str(HIGH_RANGE), str(LOW_RANGE), str(MARGIN), str(vis)
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
