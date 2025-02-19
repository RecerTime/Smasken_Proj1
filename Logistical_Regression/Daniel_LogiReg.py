#Import packages
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

#Import functions
from Logistical_Regression.functions.load_data import load_data
from Logistical_Regression.functions.missing_values_check import missing_values_check
from Logistical_Regression.functions.preprocess_data import preprocess_data
from Logistical_Regression.functions.splice_data import splice_data
from Logistical_Regression.functions.create_logistic_model import create_logistic_model
from Logistical_Regression.functions.predict_and_find_accuracy import predict_and_find_accuracy
from Logistical_Regression.functions.random_search_cv import random_search_cv

#Load data
data = load_data()
data_preprocessed = preprocess_data(data)

#Create a naive classifier that always predicts the most frequent class
naive_clf = DummyClassifier(strategy="most_frequent", random_state=42)

#Check for missing values
if not missing_values_check(data_preprocessed):
    #Splice data
    spliced_data = splice_data(data_preprocessed, 0.2)
    X_train, X_test, y_train, y_test = spliced_data[0], spliced_data[1], spliced_data[2], spliced_data[3]

    #Create and train model and naive model
    log_reg_model = create_logistic_model(X_train, y_train, 10000, 42)
    naive_clf.fit(X_train, y_train)

    #Predict and find accuracy
    accuracy, conf_matrix = predict_and_find_accuracy(log_reg_model, X_test, y_test)
    naive_accuracy, naive_conf_matrix = predict_and_find_accuracy(naive_clf, X_test, y_test)

    #Use random search to tune the model
    parameters = {
        "C": np.logspace(-4, 4, 50),
        "penalty": ["l1", "l2", "elasticnet", None],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [100, 1000, 2500, 5000, 10000]
    }

    random_search_cv = random_search_cv(log_reg_model, parameters,30)
    random_search_model = random_search_cv.fit(X_train, y_train)
    random_search_accuracy, random_search_conf_matrix = predict_and_find_accuracy(random_search_model, X_test, y_test)

    #Results
    results = {
        "Model": ["Naive", "Untuned", "Tuned"],
        "Accuracy": [naive_accuracy, accuracy, random_search_accuracy],
        "Confusion Matrix": [naive_conf_matrix, conf_matrix, random_search_conf_matrix],
    }

    results_df = pd.DataFrame(results)
    print(results_df)

else:
        print("Values missing in data")
        exit()



