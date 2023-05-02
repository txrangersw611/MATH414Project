import pandas as pd
import numpy as np
import pywt 
import scipy.io.wavfile as wavfile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
import os


def load_data(x_path): 
    #gets all the files in the directory
    files = os.listdir(x_path)

    #data is array of energy in each wave
    data = []
    labels = [None] * len(files)


    #running read function on every file in directory
    for i, file in enumerate(files):
        name = os.path.join(x_path, file)
        rate, amplitudes = wavfile.read(name)

        data.append(wavelet_transform(amplitudes))
        labels[i] = file[0:2]

    return data, labels


def split_data(data, labels):
    """
    Splits set into 80% training data and 20% testing data. Returns x_train, x_test, y_train, y_test.
    """

    #Code for testing 
    # avg_printer(labels, data)

    return train_test_split(data, labels, train_size=0.8, shuffle=True)


def train_test_folds(num_folds: int, shuffle: True, features, labels):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle)
    scores = np.empty(num_folds)

    # loop through each fold
    for i, (train_index, test_index) in enumerate(skf.split(features, labels)):
        # determine training and test sets for fold
        x_train = np.array(features)[train_index]
        y_train = np.array(labels)[train_index]
        x_test = np.array(features)[test_index]
        y_test = np.array(labels)[test_index]

        # train model
        model = tree.DecisionTreeClassifier().fit(x_train, y_train)
        #model = svm.SVC().fit(x_train.reshape(-1,1), y_train)

        # get predictions
        preds = model.predict(x_test)
        
        # get accuracy score
        score = accuracy_score(y_test, preds)
        scores[i] = score

    return scores


def wavelet_transform(data):
    #creating Daubechies wavelet object
    max_length = 10000000
    wavelet = pywt.Wavelet('db4')

    #extract coefficients from this wavelet
    coeffs = pywt.wavedec(data, wavelet) 
    
    coeffs = np.concatenate(coeffs)

    #flatten to handle 2d arrays
    if coeffs.ndim > 1:
        coeffs = coeffs.flatten()

    #truncate ones that are too long
    max_length = 100000
    if len(coeffs) > max_length:
        coeffs = coeffs[:max_length]
    
    else:
        coeffs = np.pad(coeffs, (0, max_length - len(coeffs)), mode='constant')

    return coeffs
    #coeffs = np.pad(coeffs, (0, max_length - len(coeffs)), mode='constant')
    # energy = 0
    # for coeff in coeffs:
    #     energy += np.sum(coeff ** 2)

    # return energy


def avg_printer(y, data):
    print("-------------------")
    print("Printing AVG ENERGY")

    CC = np.empty(0)
    CR = np.empty(0)
    DP = np.empty(0)
    ST = np.empty(0)
    ZM = np.empty(0)

    for i in range(len(y)):
        if y[i] == 1:
            CC = np.append(CC, data[i])
        elif y[i] == 2:
            CR = np.append(CR, data[i])
        elif y[i] == 3:
            DP = np.append(DP, data[i])
        elif y[i] == 4:
            ST = np.append(ST, data[i])
        elif y[i] == 5:
            ZM = np.append(ZM, data[i])

    print("CC AVG", np.mean(CC))
    print("CR AVG", np.mean(CR))
    print("DP AVG", np.mean(DP))
    print("ST AVG", np.mean(ST))
    print("ZM AVG", np.mean(ZM))
    
    print("-------------------")