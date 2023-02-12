# USAGE
# python3 confusionmatrix.py

# import the necessary packages
import os
import shutil
import pandas as pd
import torch
import time
import numpy as np
start = time.time()
import seaborn as sn
import matplotlib.pyplot as plt

# set the Segmentation Loss used
#loss = "Dice"
loss = "BCE"

# define the working directory
dir = os.path.join("G:/Il mio Drive/DigitalHealth/II anno I semestre/_MI/Materiale Prog MedIm/Multitask/Results",loss,"Train3/confusion matrix")

# define the pattern labels
labColumns = ['homogeneous', 'speckled', 'nucleolar', 'centromere', 'golgi', 'numem', 'mitsp']
labRows = labColumns
#labRows.append('total') # comment to obtain matrix without sum of columns

# define the intensity levels
itsColumns = ['intermediate', 'positive']
itsRows = itsColumns
#itsRows.append('total') # comment to obtain matrix without sum of columns

# define a method used to create and save all the confusion matrixes
def create_confusion_matrix(matrix, name, itsFlag = False):
  fig = plt.figure()
  # Set titles for the figure and the subplot respectively
  title = 'Intensity' if itsFlag else 'Pattern'
  fig.suptitle('Single Patch ' + title + ' Classification\nPerformances on ' + str(np.sum(matrix)) + " samples", fontsize=14, fontweight='bold')
  #colsum = np.expand_dims(np.sum(matrix, axis=0), axis=0) # comment to obtain matrix without sum of columns
  #matrix = np.append(matrix, colsum, axis=0) # comment to obtain matrix without sum of columns
  rawsum = np.expand_dims(np.sum(matrix, axis=1), axis=1)
  rawsum[rawsum == 0.0] = 1.0
  #matrix = np.append(matrix, rawsum, axis=1) # comment to obtain matrix without sum of rows
  norm = np.rint(np.multiply(np.divide(matrix, rawsum), 100)).astype(int)
  cells = np.asarray(["{0}\n{1}%".format(data, percent) for data, percent in zip(matrix.flatten(), norm.flatten())]).reshape(norm.shape)
  heatmap = sn.heatmap(norm, annot=cells, cmap="YlGnBu", fmt='',
                       xticklabels = itsRows if itsFlag else labRows,
                       yticklabels = itsColumns if itsFlag else labColumns)
  plt.ylabel('True Class')
  plt.xlabel('Predicted Class')
  plt.xticks(rotation = 45) if not itsFlag else None
  fig = heatmap.get_figure()
  plt.tight_layout()
  fig.savefig(os.path.join(dir, name+".png"))
  fig.savefig(os.path.join(dir, name+".pdf"))
  plt.close(fig)
  fig = None
  plt.clf()

# define filename prefix and empty matrix
name = "Label Confusion Matrix"
mat = np.zeros((7,7), dtype = int)

# compute the label confusion matrixes of each folds
for file in os.listdir(dir):
  if file.endswith(".npy"):
    f = os.path.join(dir, file)
    curr = np.load(f)
    if file.__contains__("label"):
      if file.__contains__("foldA"):
        create_confusion_matrix(curr, name + " foldA", False)
        mat = np.add(mat, curr)
      elif file.__contains__("foldB"):
        create_confusion_matrix(curr, name + " foldB", False)
        mat = np.add(mat, curr)
      elif file.__contains__("foldC"):
        create_confusion_matrix(curr, name + " foldC", False)
        mat = np.add(mat, curr)
      elif file.__contains__("foldD"):
        create_confusion_matrix(curr, name + " foldD", False)
        mat = np.add(mat, curr)
      elif file.__contains__("foldE"):
        create_confusion_matrix(curr, name + " foldE", False)
        mat = np.add(mat, curr)
      else:
        print("label error")

# compute the label confusion matrix of the total dataset
create_confusion_matrix(mat, name + " All", False)

print("-------------------------------------------------")

# define filename prefix and empty matrix
name = "Intensity Confusion Matrix"
mat = np.zeros((2,2),dtype=int)

# compute the label confusion matrixes of each folds and then of the total dataset
for file in os.listdir(dir):
  if file.endswith(".npy"):
    f = os.path.join(dir, file)
    curr = np.load(f)
    if file.__contains__("intensity"):
      if file.__contains__("foldA"):
        create_confusion_matrix(curr, name + " foldA", True)
        mat = np.add(mat, curr)
      elif file.__contains__("foldB"):
        create_confusion_matrix(curr, name + " foldB", True)
        mat = np.add(mat, curr)
      elif file.__contains__("foldC"):
        create_confusion_matrix(curr, name + " foldC", True)
        mat = np.add(mat, curr)
      elif file.__contains__("foldD"):
        create_confusion_matrix(curr, name + " foldD", True)
        mat = np.add(mat, curr)
      elif file.__contains__("foldE"):
        create_confusion_matrix(curr, name + " foldE", True)
        mat = np.add(mat, curr)
      else:
        print("intensity error")

# compute the label confusion matrix of the total dataset
create_confusion_matrix(mat, name + " All",  True)

# display the total time needed to run the script
print("total time: sec ", time.time() - start)
