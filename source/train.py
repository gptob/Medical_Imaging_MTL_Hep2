# USAGE
# python3 train.py

# import the necessary packages
import pandas as pd
from training.dataset import MultitaskDataset
from training.model import UNet
from training.loss import MultitaskLoss
from training import config
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
import torch.nn as nn
import sys
import torchmetrics.functional as F
import os

# set device
device = config.DEVICE
print("[INFO] device is", device)

# load the train dataset
trainData = MultitaskDataset(csv_file=config.TRAIN_DATASET_PATH, root_dir=config.IMAGE_DATASET_PATH)
numTrainSamples = int(len(trainData))
print(f"[INFO] found {numTrainSamples} samples in the training set...")
sys.stdout.flush()

# create the training data loader
trainLoader = DataLoader(trainData, shuffle=True,	batch_size=config.TRAIN_BATCH_SIZE, pin_memory=config.PIN_MEMORY,	num_workers=config.NUM_WORKERS)

# initialize our UNet model
unet = UNet(config.NUM_CHANNELS, config.NUM_CLASSES, True).to(device)

# initialize loss function and optimizer
lossFunc = MultitaskLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
trainSteps = len(trainData) // config.TRAIN_BATCH_SIZE

# initialize a dictionary to store training history
# H = {'bce_loss':[], 'ce_loss':[], 'dice_loss':[],'train_loss':[]}
History = {'mask_loss':[], 'mask_accuracy':[], 'label_loss':[], 'label_accuracy':[], 'intensity_loss':[], 'intensity_accuracy':[]}

# create training report
filename = config.MODEL_PREFIX + "training_report.txt"
f = open(filename, "w")
f.write(config.MODEL_PREFIX + "Report\n")
f.write(str(numTrainSamples) + " examples in the training set\n")
f.close()

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):

    # set the model in training mode
    unet.train()
    startEpoch = time.time()

    # initialize the training metrics
    totalMaskLoss = 0
    totalLabLoss = 0
    totalItsLoss = 0
    totalMaskAcc = 0
    totalLabAcc = 0
    totalItsAcc = 0

    # update report at the current epoch
    f = open(filename, "a")
    f.write("\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    f.write("\n---> Epoch" + str(e+1))
    f.write("\nstart time:" + str(startEpoch) + "\n")
    f.close()

    # loop over the training set
    for (i, (x, mask,label,intensity)) in enumerate(trainLoader):
 
        # update the report during the epoch
        if i % 10 == 0:
          f = open(filename, "a")
          f.write("*")
          f.close()
        
        torch.cuda.empty_cache()
        
        # send the input to the device
        x, mask, label, intensity = x.to(device),mask.to(device), label.to(device), intensity.to(device)
     
        # perform a forward pass and calculate the training loss     
        out = unet(x)
        loss = lossFunc(out, mask, label, intensity)
        currMaskLoss = loss[0].item()
        totalMaskLoss += currMaskLoss
        currLabLoss = loss[1].item()
        totalLabLoss += currLabLoss
        currItsLoss = loss[2].item()
        totalItsLoss += currItsLoss

        # compute the training accuracy
        threshold = torch.Tensor([config.THRESHOLD]).to(device)
        predMask = torch.Tensor(torch.where(torch.sigmoid(out[0]) > threshold, 1, 0)).type(torch.uint8)
        currMaskAcc = F.dice(predMask, mask.type(torch.uint8)).item()
        totalMaskAcc += currMaskAcc
        currLabAcc = torch.where(out[1].argmax(1) == label)[0].shape[0]
        totalLabAcc += currLabAcc
        predIts = torch.Tensor(torch.where(out[2].squeeze() > threshold, 1, 0)).type(torch.uint8)
        currItsAcc = torch.where(predIts == intensity)[0].shape[0]
        totalItsAcc += currItsAcc

        
        #print("mask_loss= {:.6f}, mask_acc = {:.6f}".format(currMaskLoss, currMaskAcc), end = ' ')
        #print("lab_loss= {:.6f}, lab_acc = {:.6f}".format(currLabLoss, currLabAcc), end = ' ')
        #print("Intensity Classification task: loss= {:.6f}, accuracy = {:.6f}".format(currItsLoss, currItsAcc))

        # start GradNorm code

        weighted_task_loss = torch.mul(unet.weights, loss) #model.weight * loss
        if i == 0:
            initial_task_loss = loss.data.cpu().numpy()
        total_loss = torch.sum(weighted_task_loss) #total_loss = F.mean(weighted task loss)
        opt.zero_grad(set_to_none=True) # model.cleangrads()
        total_loss.backward(retain_graph=True) # total_loss backward?)
        unet.weights.grad.data = unet.weights.grad.data * 0.0 #model.weight.cleangrad()

        W = unet.get_last_shared_layer()
        # get the gradient norms for each of the tasks
        # G^ { (i) }_W(t)
        norms = []
        for i in range(len(loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.norm(torch.mul(unet.weights[i], gygw[0])))
        norms = torch.stack(norms)

        # compute the inverse training rate r_i (t)
        loss_ratio = loss.data.cpu().numpy() / initial_task_loss
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        # compute the mean norm \tilde {G} _W(t)
        mean_norm = np.mean(norms.data.cpu().numpy())
        #print ('tilde G_W(t): {}' .format (mean_norm))

        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** config.ALPHA), requires_grad=False).cuda()
        # this is the GradNorm loss itself
        Gradloss = nn.L1Loss(reduction = 'sum')
        grad_norm_loss = 0
        for loss_index in range(0, len(loss)):
            grad_norm_loss = torch.add(grad_norm_loss, Gradloss(norms[loss_index], constant_term[loss_index]))
        # compute the gradient for the weights
        unet.weights.grad = torch.autograd.grad(grad_norm_loss,unet.weights)[0]
        
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        # opt.zero_grad()
        opt.step()
        normalize_coeff = 3 / torch.sum(unet.weights.data, dim=0)
        unet.weights.data = unet.weights.data * normalize_coeff

        # end GradNorm code

    # calculate the average training loss for each task
    avgMaskLoss = totalMaskLoss / trainSteps
    avgLabLoss = totalLabLoss / trainSteps
    avgItsLoss = totalItsLoss / trainSteps

    # calculate the average training accuracy for each task
    avgMaskAcc = totalMaskAcc / trainSteps
    avgLabAcc = totalLabAcc / numTrainSamples
    avgItsAcc = totalItsAcc / numTrainSamples
    
    # update the training history
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    History['mask_loss'].append(avgMaskLoss)
    History['label_loss'].append(avgLabLoss)
    History['intensity_loss'].append(avgItsLoss)
    History['mask_accuracy'].append(avgMaskAcc)
    History['label_accuracy'].append(avgLabAcc)
    History['intensity_accuracy'].append(avgItsAcc)

    # display the current metrics for each task
    print("Image Segmentation task: loss= {:.6f}, accuracy = {:.6f}".format(avgMaskLoss, avgMaskAcc))
    print("Pattern Classification task: loss= {:.6f}, accuracy = {:.6f}".format(avgLabLoss, avgLabAcc))
    print("Intensity Classification task: loss= {:.6f}, accuracy = {:.6f}".format(avgItsLoss, avgItsAcc))
    sys.stdout.flush()

    #save model at each epoch #too heavy
    #modelname = config.MODEL_PATH[:-4] + '_Epoch' + str(e+1) + config.MODEL_PATH[-4:]
    #torch.save(unet, modelname)

    # update the report with total time elapsed to run the current epoch
    endEpoch = time.time()
    f = open(filename, "a")
    f.write("\ntotal time: secs " + str(int(endEpoch - startEpoch)))
    f.close()
    
    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(History['mask_loss'], label='Image Segmentation')
    plt.plot(History['label_loss'], label='Pattern Classification')
    plt.plot(History['intensity_loss'], label='Intensity Classification')
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(config.PLOT_PATH)
    plt.clf()   
    plt.close()

    # plot the training accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(History['mask_accuracy'], label='Image Segmentation')
    plt.plot(History['label_accuracy'], label='Pattern Classification')
    plt.plot(History['intensity_accuracy'], label='Intensity Classification')
    plt.title("Training Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    newPath = config.PLOT_PATH[:-8] + "accuracy.png"
    plt.savefig(newPath)
    plt.clf()   
    plt.close()

    # save the current history
    table = pd.DataFrame.from_dict(History)
    csvPath = config.MODEL_PREFIX + "training_history.csv"
    table.to_csv(os.path.join(config.BASE_OUTPUT, csvPath))


# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
sys.stdout.flush()

# serialize the model to disk
torch.save(unet, config.MODEL_PATH)

