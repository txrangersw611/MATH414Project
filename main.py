import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

from data import load_data, waveletTransform, split_data


def model(file):
    #load the Data
    sample_rate, data = load_data("data.csv")

    #split into train and test data
    train_x, train_y, test_x, test_y = split_data(data)

    #might need this to anylize the data with wavelets
    processed_x_train = waveletTransform(train_x)
    processed_x_test = waveletTransform(test_x)


    #train and test the model
    model = tree.DecisionTreeClassifier() #need to decide what model were gonna do (how are we gonna include wavelts)
    model.fit(processed_x_train, train_y)

    #get accuracy from model
    preds = model.predict(processed_x_test)
    score = accuracy_score(test_y, preds)

    #print results
    print("The accuracy of the model is", score)

    #######################################################
    #testing for the inputted file



if __name__ == "__main__":
    file = input("Enter bird call file name")
    model(file)

