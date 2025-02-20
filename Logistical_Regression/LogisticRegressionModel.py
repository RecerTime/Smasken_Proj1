# Import packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, ConfusionMatrixDisplay

# Import functions
from Logistical_Regression.functions.load_data import load_data
from Logistical_Regression.functions.missing_values_check import missing_values_check
from Logistical_Regression.functions.preprocess_data import preprocess_data
from Logistical_Regression.functions.splice_data import splice_data
from Logistical_Regression.functions.create_logistic_model import create_logistic_model
from Logistical_Regression.functions.predict_and_find_accuracy import predict_and_find_accuracy
from Logistical_Regression.functions.random_search_cv import random_search_cv

# Load data
data = load_data()
data_preprocessed = preprocess_data(data)

# Create a naive classifier that always predicts the most frequent class
naive_clf = DummyClassifier(strategy="most_frequent", random_state=42)

# Check for missing values
if not missing_values_check(data_preprocessed):
    # Splice data
    spliced_data = splice_data(data_preprocessed, 0.2)
    X_train, X_test, y_train, y_test = spliced_data[0], spliced_data[1], spliced_data[2], spliced_data[3]

    # Create and train model and naive model
    log_reg_model = create_logistic_model(X_train, y_train, 10000, 42)
    naive_clf.fit(X_train, y_train)

    # Predict and find accuracy
    accuracy, conf_matrix = predict_and_find_accuracy(log_reg_model, X_test, y_test)
    naive_accuracy, naive_conf_matrix = predict_and_find_accuracy(naive_clf, X_test, y_test)

    # Use random search to tune the model
    parameters = {
        "C": np.logspace(-4, 4, 50),
        "penalty": ["l1", "l2", "elasticnet", None],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [100, 1000, 2500, 5000, 10000]
    }

    random_search_cv = random_search_cv(log_reg_model, parameters, 100)
    tuned_model = random_search_cv.fit(X_train, y_train)
    tuned_model_accuracy, tuned_model_conf_matrix = predict_and_find_accuracy(tuned_model, X_test, y_test)

    untuned_precision = precision_score(y_test, log_reg_model.predict(X_test))
    untuned_recall = recall_score(y_test, log_reg_model.predict(X_test))
    untuned_f1 = f1_score(y_test, log_reg_model.predict(X_test))

    tuned_precision = precision_score(y_test, tuned_model.predict(X_test))
    tuned_recall = recall_score(y_test, tuned_model.predict(X_test))
    tuned_f1 = f1_score(y_test, tuned_model.predict(X_test))

    naive_clf_precision = precision_score(y_test, naive_clf.predict(X_test))
    naive_clf_recall = recall_score(y_test, naive_clf.predict(X_test))
    naive_clf_f1 = f1_score(y_test, naive_clf.predict(X_test))

    fpr_untuned, tpr_untuned, thresholds_untuned = roc_curve(y_test, log_reg_model.predict_proba(X_test)[:, 1])
    fpr_tuned, tpr_tuned, thresholds_tuned = roc_curve(y_test, tuned_model.predict_proba(X_test)[:, 1])
    fpr_naive, tpr_naive, thresholds_naive = roc_curve(y_test, naive_clf.predict_proba(X_test)[:, 1])
    plt.plot(fpr_untuned, tpr_untuned, label="Untuned")
    plt.plot(fpr_tuned, tpr_tuned, label="Tuned")
    plt.plot(fpr_naive, tpr_naive, label="Naive", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    naive_conf_matrix_disp = ConfusionMatrixDisplay(naive_conf_matrix,display_labels=[0, 1])
    untuned_conf_matrix_disp = ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1])
    tuned_conf_matrix_disp = ConfusionMatrixDisplay(tuned_model_conf_matrix,display_labels=[0, 1])
    naive_conf_matrix_disp.plot()
    untuned_conf_matrix_disp.plot()
    tuned_conf_matrix_disp.plot()
    plt.show()

    print(random_search_cv.best_params_)
    # Results
    results = {
        "Model": ["Naive", "Untuned", "Tuned"],
        "Accuracy": [naive_accuracy, accuracy, tuned_model_accuracy],
        "Confusion Matrix": [naive_conf_matrix, conf_matrix, tuned_model_conf_matrix],
        "Precision": [naive_clf_precision, untuned_precision, tuned_precision],
        "Recall": [naive_clf_recall, untuned_recall, tuned_recall],
        "F1": [naive_clf_f1, untuned_f1, tuned_f1]
    }

    results_df = pd.DataFrame(results)
    print(results_df)

else:
    print("Values missing in data")
    exit()
