# import the necessary packages
import torch
import os

# initialize step of cross validation
CV_STEP = 1

# initialize GradNorm parameter
ALPHA = 0.06

DICE = True # if True to set Dice as Segmentation Loss
            # if False to set BCE as Segmentation Loss

# define the folds, train and test sets for each cv-step
FOLD_NAMES = ["foldA", "foldB", "foldC", "foldD", "foldE"]
TRAIN_NAMES = [i+'_train.csv' for i in FOLD_NAMES]
TEST_NAMES = [i+'_test.csv' for i in FOLD_NAMES]
TRAIN_DATASET = TRAIN_NAMES[CV_STEP]
TEST_DATASET = TEST_NAMES[CV_STEP]

# initialize model name prefix
MODEL_PREFIX = "Train2_Dice_" + FOLD_NAMES[CV_STEP] + "_"

# set number of workers on the device
NUM_WORKERS = 1

# define paths to image samples, train and test datasets
IMAGE_DATASET_PATH = os.path.join("dataset", "train")
MASK_DATASET_PATH = os.path.join("dataset", "final_patches.csv")
TRAIN_DATASET_PATH = os.path.join('dataset', 'crossValidationTables', 'train', TRAIN_DATASET)
TEST_DATASET_PATH = os.path.join('dataset', 'crossValidationTables', 'test', TEST_DATASET)


# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 1
NUM_LEVELS = 3

# initialize learning rate and the number of epochs to train for
INIT_LR = 0.0001
NUM_EPOCHS = 35

# initialize train and test batch size
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 1

# define the input image dimensions
INPUT_IMAGE_WIDTH = 384
INPUT_IMAGE_HEIGHT = 384

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model
MODEL_NAME = MODEL_PREFIX + "UNet.pth"
MODEL_PATH = os.path.join(BASE_OUTPUT, MODEL_NAME)

# define the path to the model training plot
PLOT_NAME = MODEL_PREFIX + "plot_train_loss.png"
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, PLOT_NAME])
