from sklearn.metrics import confusion_matrix

def predict_and_find_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, conf_matrix
