# USAGE
# python3 predict.py

# import the necessary packages
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from training.dataset import MultitaskDataset
from training import config
import torch
import torch.nn as nn
import os
import sys
import time
import torchmetrics.functional as F
from sklearn.metrics import confusion_matrix

start = time.time()

# set the device we will be using to train the model
device = config.DEVICE
print("[INFO] device is", device)

# load the test dataset
print("[INFO] loading the test dataset...")
testData = MultitaskDataset(csv_file=config.TEST_DATASET_PATH, root_dir=config.IMAGE_DATASET_PATH)
numTestSamples = int(len(testData))
print(f"[INFO] found {numTestSamples} examples in the test set...")
sys.stdout.flush()

# initialize the test data loader
testLoader = DataLoader(testData, batch_size=config.TEST_BATCH_SIZE, pin_memory=config.PIN_MEMORY,	num_workers=config.NUM_WORKERS)

# initialize a dictionary to store testing history
History = {'dice_score':[], 'label_true':[], 'label_predicted':[], 'intensity_true':[], 'intensity_predicted':[]}
totalMaskAcc = 0
totalLabAcc = 0
totalItsAcc = 0

# define useful variables to make predictions
threshold = torch.Tensor([config.THRESHOLD]).to(device)
softmax = nn.Softmax(dim=1)

# load the model and set it to evaluation mode
model = torch.load(config.MODEL_PATH).to(device)
model.eval()

# switch off autograd
with torch.no_grad():
    # loop over the test set
    for (i, (image, mask, label, intensity)) in enumerate(testLoader):
          
        # send the input to the device and make predictions on it
        image, mask = image.to(device),mask.to(device)
        out = model(image)

        # compute the image segmentation accuracy
        predMask = torch.Tensor(torch.where(torch.sigmoid(out[0]) > threshold, 1, 0)).type(torch.uint8)
        diceScore = F.dice(predMask, mask.type(torch.uint8)).item()
        #print('diceScore: ', diceScore, end = ' - ')
        History['dice_score'].append(diceScore) # import torchmetrics.functional as F
        totalMaskAcc += diceScore
        #print('totalMaskAcc: ', totalMaskAcc)

        # compute the pattern classification accuracy
        predLabel = softmax(out[1]).argmax(1).item()
        #print('predLabel: ', predLabel, end = ' - ')
        History['label_predicted'].append(predLabel)
        #print('trueLabel:', label, end = ' - ')
        History['label_true'].append(label.item())
        totalLabAcc += 1 if predLabel == label.item() else 0
        #print('totalLabAcc: ', totalLabAcc)

        # compute the intensity classification accuracy
        predIntensity = torch.Tensor(torch.where(out[2].squeeze() > threshold, 1, 0)).type(torch.uint8).item()
        #print('predIntensity: ', predIntensity, end = ' - ')
        History['intensity_predicted'].append(predIntensity)
        #print('trueIntensity:', intensity, end = ' - ')
        History['intensity_true'].append(intensity.item())
        totalItsAcc += 1 if predIntensity == intensity.item() else 0
        #print('totalItsAcc: ', totalItsAcc)
        
    # create test report and add to it the model performances
    filename = config.MODEL_PREFIX + "test_report.txt"
    f = open(filename, "w")
    f.write(config.MODEL_PREFIX + "Test Report\n")
    f.write(str(numTestSamples) + " samples in the test set")
    f.write("\ndice score min value:" + str(min(History['dice_score'])))
    f.write("\ndice score mean value:" + str(totalMaskAcc / numTestSamples))
    f.write("\ndice score max value:" + str(max(History['dice_score'])))
    f.write("\n label classification accuracy %: " + str(totalLabAcc / numTestSamples))
    f.write("\n intensity classification accuracy %: " + str(totalItsAcc / numTestSamples))
    f.close()
    print(f"[INFO] test report created")
    
    # save the test history
    table = pd.DataFrame.from_dict(History)
    csvPath = config.MODEL_PREFIX + "test_history.csv"
    table.to_csv(os.path.join(config.BASE_OUTPUT, csvPath),index=False)
    print(f"[INFO] performance saved")
    
    # compute confusion matrix for the pattern classification task
    labelMatrix = np.array(confusion_matrix(History['label_true'], History['label_predicted']))
    filename = config.MODEL_PREFIX + "labelMatrix.npy"
    np.save(os.path.join(config.BASE_OUTPUT, filename), labelMatrix)
    
    # compute confusion matrix for the intensity classification task
    intensityMatrix = np.array(confusion_matrix(History['intensity_true'], History['intensity_predicted']))
    filename = config.MODEL_PREFIX + "intensityMatrix.npy"
    np.save(os.path.join(config.BASE_OUTPUT, filename), intensityMatrix)
    
   # display the total time needed to perform the testing
    time = time.time() - start
    print(f"[INFO] test routine completed in {time:.2f} seconds")
    


