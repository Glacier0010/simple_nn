# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np

def load_CIFAR10(ROOT):     #加载cifar10的所有数据
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        with open(f, "rb") as ff:
            datadict = pickle.load(ff, encoding="latin1")
            X = datadict["data"]
            Y = datadict["labels"]
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)        
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y

    with open(os.path.join(ROOT, "test_batch"), "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        Xte = datadict["data"]
        Yte = datadict["labels"]
        Xte = Xte.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Yte = np.array(Yte)   
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=5000, num_validation=1000, num_test=1000, subtract_mean=True):
    """
    加载数据，并进行初步处理
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(os.path.dirname(__file__), "datasets/cifar-10-batches-py")
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
