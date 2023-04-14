import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def load_data(x_path): 
    return pd.read_csv(x_path)


def split_data(data):
    y = data.loc[-1,:] #just putting some garbo here (need to get the labels)
    x = data.drop[-1] #getting features
        
    return train_test_split(x, y, test_size=.8)


def preprocess_x(data):
    #something with wavelets
    return data

