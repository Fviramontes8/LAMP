#!/usr/bin/env python
# coding: utf-8


import scipy.io as sio
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import pickle


def load_train(data1):
    """
    Loads the data from the matfile
    Performs normalization of the data using the maximum and minimum value
    Extracts the train data and labels into X, y
    """

    train = data1[
        0: round(data1.shape[0] / 2), :
    ]   # taking first half of the year for training
    max1 = train[:, 1:7].max(axis=0)   # computing the maximum of train data
    min1 = train[:, 1:7].min(axis=0)   # computing the minimum of train data
    train[:, 1:7] = (train[:, 1:7] - min1) / (max1 - min1)   # Normalization

    train_f = train[np.where(train[:, 10] > 0)]   # getting only the fault data
    lab_f = train_f[:, 10]  # getting the corresponding labels

    xtrain = train_f
    ytrain = lab_f
    X = xtrain[1::120, 1:7]   # subsampling by a factor of 120
    y = ytrain[1::120]    # subsampling the labels

    return X, y, max1, min1


def splitting_train_val(X, y):
    """
    Doing a stratified 2 fold split for splitting
    the data into train and validation set
    """

    skf = StratifiedKFold(n_splits=2)
    skf.get_n_splits(X, y)

    for train_index, val_index in skf.split(X, y):
        # print("TRAIN:", train_index, "VAL:", val_index)
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

    return x_train, y_train


def loading_matfile_data(relay, mat_file):
    os.chdir('../Data')
    data1 = sio.loadmat(relay + '.mat')[mat_file]
    [X, y, max_tr, min_tr] = load_train(data1)
    [X_train, y_train] = splitting_train_val(X, y)
    return X_train, y_train, max_tr, min_tr


def train_svm(X, y):
    svm_model = SVC(kernel='linear', C=100).fit(X, y)
    return svm_model


def main():
    relay = 'RTL3'
    configs = [
        'C1',
        'C2',
        'C3',
        'C4',
    ]

    for config in configs:
        mat_file = config + relay

        [X, y, max_tr, min_tr] = loading_matfile_data(relay, mat_file)
        svm_model = train_svm(X, y)

        dir_sav = '../Models'
        file_name = f'{mat_file}_svmmodel.sav'
        os.chdir(dir_sav)
        pickle.dump(svm_model, open(file_name, 'wb'))
        np.save(f'maxmin_svm_{mat_file}.npy', [max_tr, min_tr])


if __name__ == '__main__':
    main()
