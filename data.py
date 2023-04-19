import pandas as pd
import numpy as np
import pywt 
import scipy.io.wavfile as wavfile
from sklearn.model_selection import train_test_split
import os

def load_data(x_path): 
    #gets all the files in the directory
    files = os.listdir(x_path)

    #data is 3D array, each index is the wav files transformed into a 2D array of amplitudes (I think)
    data = np.empty(0) 

    # name = os.path.join(x_path, "ZM08.wav")
    # rate, data = wavfile.read(name)
    # print(data.shape)

    #running read function on every file in directory
    for file in files:
        name = os.path.join(x_path, file)
        rate, amplitudes = wavfile.read(name)
        print(amplitudes)
        data = np.append(data, amplitudes)
        print(data)
        print("-------------------")


    return data


def split_data(data):
    y = np.empty(100)

    #making y filled with 10 1'2, 10 2's, etc
    for i in range(100):
        y[i] = i/10
        print(i/10)

    print(y)
    return train_test_split(data, y, test_size=.8, shuffle=True)


def waveletTransform(data):
    #creating Daubechies wavelet objext
    wavelet = pywt.Wavelet('db4')

    #extract coefficients from this wavelet
    coeffs = pywt.wavedec(data, wavelet, level=6) #doing 6 because thats what the research paper did
    
    features = np.sum(coeffs**2)
    return features

