import itertools

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import tree

from data import load_data, preprocess_x, split_data


def main():
    #load the Data
    data = load_data("data.csv")

    #split into train and test data
    train_x, train_y, test_x, test_y = split_data(data)

    #might need this to anylize the data with wavelets
    processed_x_train = preprocess_x(train_x)
    processed_x_test = preprocess_x(test_x)


    #train and test the model
    model = tree.DecisionTreeClassifier() #need to decide what model were gonna do (how are we gonna include wavelts)
    model.fit(processed_x_train, train_y)


    #get accuracy from model


    #print results



if __name__ == "__main__":
    main()

