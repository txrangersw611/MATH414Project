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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


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


def train_test_folds(num_folds: int, shuffle: True, features, labels):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle)
    scores = np.zeros(num_folds)

    # loop through each fold
    for i, (train_index, test_index) in enumerate(skf.split(features, labels)):
        # determine training and test sets for fold
        x_train = np.array(features)[train_index]
        y_train = np.array(labels)[train_index]
        x_test = np.array(features)[test_index]
        y_test = np.array(labels)[test_index]

        # train model
        model = RandomForestClassifier().fit(x_train, y_train)
        #model = svm.SVC().fit(x_train.reshape(-1,1), y_train)

        # get predictions
        preds = model.predict(x_test)
        
        # get accuracy score
        score = accuracy_score(y_test, preds)
        scores[i] = score

        print(scores)

    return scores


def wavelet_transform(data):
    #creating Daubechies wavelet object
    wavelet = pywt.Wavelet('db4')

    #extract coefficients from this wavelet
    coeffs = pywt.wavedec(data, wavelet) 

    coeffs = np.concatenate(coeffs)

    
    #flatten if needed
    if coeffs.ndim > 1:
        coeffs = coeffs.flatten()

    #truncate samples that are too long
    max_length = 5000000
    if len(coeffs) > max_length:
        coeffs = coeffs[:max_length]
    
    else:
        coeffs = np.pad(coeffs, (0, max_length - len(coeffs)), mode='constant')

    return coeffs
