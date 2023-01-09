# USAGE
# python3 train.py

# import the necessary packages
from training.dataset import MultitaskDataset
from training.model import UNet
from training.loss import MultitaskLoss
from training import config
from torch.utils.data import random_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os
import numpy as np
import torch.nn as nn
import sys

# load the image and mask filepaths in a sorted manner
root_dir = config.IMAGE_DATASET_PATH
csv_file = config.TRAIN_DATASET_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device is", device)

# create the train and test datasets
trainData = MultitaskDataset(csv_file=csv_file, root_dir=root_dir)

#print(trainData)
numTrainSamples = int(len(trainData))
print(f"[INFO] found {len(trainData)} examples in the training set...")
sys.stdout.flush()

# create the training data loaders
trainLoader = DataLoader(trainData, shuffle=True,	batch_size=config.TRAIN_BATCH_SIZE, pin_memory=config.PIN_MEMORY,	num_workers=config.NUM_WORKERS)

# initialize our UNet model
unet = UNet(config.NUM_CHANNELS, config.NUM_CLASSES, True).to(config.DEVICE)

# initialize loss function and optimizer

lossFunc = MultitaskLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)

# calculate steps per epoch for training and test set
trainSteps = len(trainData) // config.TRAIN_BATCH_SIZE

# initialize a dictionary to store training history
H = {'bce_loss':[], 'ce_loss':[], 'dice_loss':[],'train_loss':[]}

f = open("report.txt", "w")
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

    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalDiceLoss = 0
    totalCELoss = 0
    totalBCELoss = 0
    
    cnt = 0
    f = open("report.txt", "a")
    f.write("\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    f.write("\n---> Epoch" + str(e+1))
    f.write("\nstart time:" + str(startEpoch) + "\n")
    f.close()

    # loop over the training set
    for (i, (x, mask,label,intensity)) in enumerate(trainLoader):
 
        cnt += 1
        if cnt == 10:
          f = open("report.txt", "a")
          f.write("*")
          f.close()
          cnt = 0
        
        torch.cuda.empty_cache()
        
        # send the input to the device
        x, mask, label, intensity = x.to(config.DEVICE),mask.to(config.DEVICE), label.to(config.DEVICE), intensity.to(config.DEVICE)
     
        # perform a forward pass and calculate the training loss     
        out = unet(x)
        loss  = lossFunc(out, mask, label, intensity)

        totalDiceLoss += loss[0].item()
        totalCELoss += loss[1].item()
        totalBCELoss += loss[2].item()

        weighted_task_loss = torch.mul(unet.weights, loss) #model.weight * loss
        if i == 0:
            initial_task_loss = loss.data.cpu().numpy()
        total_loss = torch.sum(weighted_task_loss) #total_loss = F.mean(weighted task loss)
        opt.zero_grad(set_to_none=True) # model. cleangrads ()
        total_loss.backward(retain_graph=True) # total_ loss backward?)
        unet.weights.grad.data = unet.weights.grad.data * 0.0 #model.weight.cleangrad()

        W = unet.get_last_shared_layer()
        # get the gradient norms for each of the tasks
        # G^ { (i) }_W(t)
        norms = []
        for i in range(len(loss)):
            # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(loss[i], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append (torch.norm(torch.mul(unet.weights[i], gygw[0])))
        norms = torch.stack(norms)

        # compute the inverse training rate r_i (t)
        loss_ratio = loss.data.cpu().numpy() / initial_task_loss
        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        # compute the mean norm \tilde {G} _W(t)
        mean_norm = np.mean(norms.data.cpu().numpy())
        #print ('tilde G_W(t): {}' .format (mean_norm))

        # compute the GradNorm loss
        # this term has to remain constant
        constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 0.06), requires_grad=False).cuda()
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
        # add the loss to the total training loss so far
    
    
    totalTrainLoss = totalDiceLoss + totalCELoss + totalBCELoss
    
    # calculate the average training loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgDiceLoss = totalDiceLoss / trainSteps
    avgCELoss = totalCELoss / trainSteps
    avgBCELoss = totalBCELoss / trainSteps
    
    # update our training history
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    
    print("Dice loss: {:.6f}".format(avgDiceLoss), end = ' - ')
    H['dice_loss'].append(avgDiceLoss)
    
    print("Cross Entropy: {:.6f}".format(avgCELoss), end = ' - ')
    H['ce_loss'].append(avgCELoss)

    print("Binary Cross Entropy: {:.6f}".format(avgBCELoss))
    H['bce_loss'].append(avgBCELoss)
    
    print("Train loss: {:.6f}".format(avgTrainLoss))
    H['train_loss'].append(avgTrainLoss)
    
    sys.stdout.flush()
    
    #save variables at each epoch
    modelname = config.MODEL_PATH[:-4] + '_Epoch' + str(e+1) + config.MODEL_PATH[-4:]
    torch.save(unet, modelname)
    # add toreport file current loss values
    endEpoch = time.time()
    f = open("report.txt", "a")
    f.write("\nend time:" + str(endEpoch))
    f.write("\ntotal time: secs " + str(int(endEpoch - startEpoch)))
    f.write("\nsegmentation loss: " + str(avgDiceLoss))
    f.write("\npattern classification loss: " + str(avgCELoss))
    f.write("\nintensity classification loss: " + str(avgBCELoss))
    f.write("\ntotal train loss: " + str(avgTrainLoss))
    f.close()
    


# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
sys.stdout.flush()

# serialize the model to disk
torch.save(unet, config.MODEL_PATH)

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H['dice_loss'], label='Dice Loss (Image Segmentation)')
plt.plot(H['ce_loss'], label='Cross Entropy Loss (Pattern Classification)')
plt.plot(H['bce_loss'], label='Binary Cross Entropy Loss (Intensity Classification)')
plt.plot(H['train_loss'], label='Train loss (Total)')
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig(config.PLOT_PATH)

