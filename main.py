import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn import tree
from data import load_data, wavelet_transform, split_data, train_test_folds


def model():
    #load the Data (data is an array of all the energies of each wave)
    data, labels = load_data("data")

    #split into train and test data
    train_x, test_x, train_y, test_y = split_data(data, labels)

    # #train and test the model
    # model = tree.DecisionTreeClassifier() #need to decide what model to use & how to incorporate wavelets
    # model.fit(train_x.reshape(-1,1), train_y)

    # #get accuracy from model
    # preds = model.predict(test_x.reshape(-1,1))
    # score = accuracy_score(test_y, preds)

    scores = train_test_folds(5, True, data, labels)
    print(scores)
    print(max(scores))

    #print results
    # print("The accuracy of the model is", score)

    #######################################################
    #testing for the inputted file
    #load the Data
    #sample_rate, data = load_data("data.csv")


if __name__ == "__main__":
    #file = input("Enter bird call file name")
    model()

