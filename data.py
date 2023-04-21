import pandas as pd
import numpy as np
import pywt 
import scipy.io.wavfile as wavfile
from sklearn.model_selection import train_test_split
import os

def load_data(x_path): 
    #gets all the files in the directory
    files = os.listdir(x_path)

    #data is array of energy in each wave
    data = np.empty(58)

    #running read function on every file in directory
    for i, file in enumerate(files):
        name = os.path.join(x_path, file)
        rate, amplitudes = wavfile.read(name)
        data[i] = waveletTransform(amplitudes)

    return data


def split_data(data):
    """
    CC is NORTHERN CARDINAL
    CR is BLUE JAY
    DP is DOWNY WOODPECKER
    ST is AMERICAN GOLDFINCH
    ZM is MOURNING DOVE
    """
    y = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
         2,2,2,2,2,2,2,2,2,2,2,2,
         3,3,3,3,3,3,3,3,3,3,
         4,4,4,4,4,4,4,4,4,4,
         5,5,5,5,5,5,5,5,5,5]
    y = np.array(y)

    #Code for testing 
    avg_printer(y, data)

    return train_test_split(data, y, train_size=.8, shuffle=True)


def waveletTransform(data):
    #creating Daubechies wavelet objext
    wavelet = pywt.Wavelet('db4')

    #extract coefficients from this wavelet
    coeffs = pywt.wavedec(data, wavelet, level=6) #doing 6 because thats what the research paper did
    
    energy = 0
    for coeff in coeffs:
        energy += np.sum(coeff ** 2)

    return energy

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







