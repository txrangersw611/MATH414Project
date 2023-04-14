import pandas as pd
import numpy as np
import pywt 
import scipy.io.wavfile as wavfile
from sklearn.model_selection import train_test_split

def load_data(x_path): 
    return wavfile.read(x_path)


def split_data(data):
    y = np.empty()
    x = data.drop[-1] #getting features

    return train_test_split(x, y, test_size=.8)


def waveletTransform(data):
    #creating Daubechies wavelet objext
    wavelet = pywt.Wavelet('db4')

    #extract coefficients from this wavelet
    coeffs = pywt.wavedec(data, wavelet, level=6) #doing 6 because thats what the research paper did
    
    features = np.sum(coeffs**2)
    return features

