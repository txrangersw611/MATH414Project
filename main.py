import pandas as pd
import numpy as np
from data import load_data, train_test_folds
from sklearn.naive_bayes import GaussianNB

def model():
    #load data
    data, labels = load_data("data2")

    # model_pipeline = []
    # model_pipeline.append(LogisticRegression(solver='liblinear'))
    # model_pipeline.append(SVC())
    # model_pipeline.append(KNeighborsClassifier())
    # model_pipeline.append(DecisionTreeClassifier())
    # model_pipeline.append(RandomForestClassifier())
    # model_pipeline.append(GaussianNB())

    # model_list = ['Logistic Regression', 'SVM', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes']
    # accuracy_list = []
    # std_list = []

    # for model in model_pipeline:
    #     scores = train_test_folds(10, True, data, labels, model)
    #     print(scores)
    #     accuracy_list.append(np.average(scores))
    #     std_list.append(np.std(scores))

    model = GaussianNB()
    scores = train_test_folds(10, True, data, labels, model)
    print(scores)
    print("Accuracy: %.3f (%.3f)" % (np.average(scores), np.std(scores)))

    # results_df = pd.DataFrame({'Model':model_list, 'Accuracy':accuracy_list, 'Std. Dev.':std_list})
    # print(results_df)
    # print()


if __name__ == "__main__":
    model()

