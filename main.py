import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

from data import load_data, waveletTransform, split_data


def model():
    #load the Data (data is a array of all the energies of each wave)
    data = load_data("data")


    #split into train and test data
    train_x, test_x, train_y, test_y = split_data(data)


    #train and test the model
    model = tree.DecisionTreeClassifier() #need to decide what model were gonna do (how are we gonna include wavelts)
    model.fit(train_x.reshape(-1,1), train_y)


    #get accuracy from model
    preds = model.predict(test_x.reshape(-1,1))
    score = accuracy_score(test_y, preds)


    #print results
    print("The accuracy of the model is", score)


    #######################################################
    #testing for the inputted file
    #load the Data
    #sample_rate, data = load_data("data.csv")


if __name__ == "__main__":
    #file = input("Enter bird call file name")
    model()

