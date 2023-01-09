import numpy
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
from torchmetrics import Dice
from training import config
 

class MultitaskLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MultitaskLoss, self).__init__()
    
    def _dice_loss(self, inputs, targets,  eps=1e-7):

        targets = targets.type(torch.int64).to(config.DEVICE)
        #print('targets',targets.shape, end=' - ')
        
        true_1_hot = torch.eye(2, device=config.DEVICE)[targets.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        
        inputs = torch.sigmoid(inputs)
        #print('inputs',inputs.shape, end=' - ')
        
        ninputs = 1 - inputs
        probas = torch.cat([inputs, ninputs], dim=1)
        true_1_hot = true_1_hot.type(inputs.type())
        dims = (0,) + tuple(range(2, targets.ndimension()))
        
        #print('probas',probas.shape, end=' - ')
        #print('true_1_hot',true_1_hot.shape, end=' - ')
        
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        loss = (1 - dice_loss)
        
        return loss
        
        
    def forward(self, preds, mask, label, intensity):

        #diceLoss = Dice().to(config.DEVICE)
        #bceWithLogits = BCEWithLogitsLoss()
        softmax = nn.Softmax(dim=1)
        crossEntropy = nn.CrossEntropyLoss()
        binaryCrossEntropy = nn.BCEWithLogitsLoss()
        label = label.long()
        mask = mask.int()
        intensity = intensity.unsqueeze(1)
        intensity = intensity.float()
        #segLoss = diceLoss(preds[0], mask)
        #segLoss = bceWithLogits(preds[0], mask)
        
        predMask, predLab, predIts = preds[0], preds[1], preds[2]
        #print('predMask',predMask.shape)
        #print('predLab',predLab.shape) 
        #print('predIts',predIts.shape) 

        segLoss = self._dice_loss(predMask, mask)
        predLab = softmax(predLab)
        labLoss = crossEntropy(predLab, label)
        itsLoss = binaryCrossEntropy(predIts, intensity)
        
        return torch.stack([segLoss, labLoss, itsLoss])
        
