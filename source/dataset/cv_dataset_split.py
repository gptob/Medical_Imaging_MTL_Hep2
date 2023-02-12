#!/usr/bin/env python3
import random as rd
#import numpy as np
import pandas as pd
import os

#GLOBAL VARIABLES
items = 252
dataset_dir = os.getcwd()  # images are in the '/train' subdirectory
array = [] # used to manage the five folds during the division of the patients


def choose_fold(iteration):
    global array
    value = iteration % 5
    if value == 0:
        array = ['a', 'b', 'c', 'd', 'e']
        rd.shuffle(array)
    # print("iteration: ", iteration)
    curr = rd.randint(0, len(array) - 1)
    fold = array[curr]
    # print("fold: ", fold)
    array.pop(curr)
    # print("array: ", array)
    return fold


def create_folds(items):
    folds = dict()
    rd.seed(12357)
    patients = [i for i in range(1, items + 1)]
    rd.shuffle(patients)

    for i in range(items):
        curr = rd.randint(0, items - i - 1)
        # print("---curr patient: ", patients[curr])
        folds[patients[curr]] = choose_fold(i)
        patients.pop(curr)

    return organize_folds(folds)


def organize_folds(folds):
    a, b, c, d, e = [], [], [], [], []

    for elem in folds.items():
        if elem[1] == 'a':
            a.append(elem[0])
        elif elem[1] == 'b':
            b.append(elem[0])
        elif elem[1] == 'c':
            c.append(elem[0])
        elif elem[1] == 'd':
            d.append(elem[0])
        elif elem[1] == 'e':
            e.append(elem[0])

    a.sort()
    print('a fold has %d items :' % len(a), a)
    b.sort()
    print('b fold has %d items :' % len(b), b)
    c.sort()
    print('c fold has %d items :' % len(c), c)
    d.sort()
    print('d fold has %d items :' % len(d), d)
    e.sort()
    print('e fold has %d items :' % len(e), e)

    return (tuple(a), tuple(b), tuple(c), tuple(d), tuple(e))


def select_indexes(images, prefix, indexes):
    i = 0
    for image in images:
        if image.startswith(prefix):
            indexes += [i,]
        i += 1
    return indexes


if __name__ == '__main__':

    folds = create_folds(items)  # return a tuple of tuples that represents the patients division

    final_patches = pd.read_csv(os.path.join(dataset_dir, "final_patches.csv"), names=["Image", "Mask", "Label", "Intensity"])
    samples = final_patches['Image'].tolist()

    fold_names = ["foldA", "foldB", "foldC", "foldD", "foldE"]
    train_names = [i+'_train.csv' for i in fold_names]
    test_names = [i+'_test.csv' for i in fold_names]

    csvs_dir = os.path.join(dataset_dir, 'crossValidationTables')
    os.mkdir(csvs_dir)
    test_dir = os.path.join(csvs_dir, 'test')
    os.mkdir(test_dir)
    train_dir = os.path.join(csvs_dir, 'train')
    os.mkdir(train_dir)

    for fold in folds:
        indexes = []
        for patient in fold:
            select_indexes(samples, f'{patient:05d}', indexes)

        test_set = final_patches.iloc[indexes]
        # save to a new DataFrame the samples corresponding to a given fold of patients using their indexes
        train_set = final_patches.drop(indexes, inplace=False)
        # save to a new DataFrame the residual samples starting from final_patches and dropping the selected indexes

        test_set.to_csv(os.path.join(test_dir, test_names[folds.index(fold)]), header=False, index=False)
        train_set.to_csv(os.path.join(train_dir, train_names[folds.index(fold)]), header=False, index=False)
