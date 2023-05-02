import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from data import load_data, split_data, train_test_folds
from sklearn import tree
from sklearn.metrics import accuracy_score

def model():
    #load data
    data, labels = load_data("data2")

    scores = train_test_folds(10, True, data, labels)
    print(np.average(scores))


if __name__ == "__main__":
    #file = input("Enter bird call file name")
    model()

