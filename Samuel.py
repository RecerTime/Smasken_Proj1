import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.metrics as skl_me
import sklearn.model_selection as skl_ms
import Preprocessing

path = "training_data_vt2025.csv"
df = pd.read_csv(path, na_values="?", dtype={"ID": str}).dropna()
data = Preprocessing.Preprocess(df)
X = data.drop(columns=['increase_stock'])
y = data['increase_stock']

X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, y, test_size=0.2, random_state=42)
n_fold = 500
K = np.arange(1,40)
cv = skl_ms.KFold(n_splits=n_fold, shuffle=True, random_state=42)
missclassification = np.zeros(len(K))

for train_index, val_index in cv.split(X_train):
    X_train_sub, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_sub, y_val = y_train.iloc[train_index], y_train.iloc[val_index] 
    for j, k in enumerate(K):
        model = skl_nb.KNeighborsClassifier(n_neighbors=k, n_jobs=4)
        model.fit(X_train_sub, y_train_sub)
        prediction = model.predict(X_val)
        missclassification[j] += np.mean(prediction != y_val)
missclassification /= n_fold

#plt.title('Cross Validation Error for kNN')
#plt.xlabel('k')
#plt.ylabel('Validation Error')
#plt.plot(K,missclassification,'.')
#plt.show()

k = K[np.where(missclassification==missclassification.min())][0]
model = skl_nb.KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
missclassification = np.mean(prediction != y_test)
#print(k)
#print(missclassification)

#confusion_matrix = skl_me.confusion_matrix(y_test, prediction)
#cm = skl_me.ConfusionMatrixDisplay(confusion_matrix)
#cm.plot(cmap='Blues')
#plt.show()

print(skl_me.classification_report(y_test, prediction))


'''
path = "training_data_vt2025.csv"
df = pd.read_csv(path, na_values="?", dtype={"ID": str}).dropna()
data = Preprocessing.Preprocess(df)
X = data.drop(columns=['increase_stock'])
y = data['increase_stock']

X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, y, test_size=0.2, random_state=42)
n_fold = 500
K = np.arange(1,40)
parameters = {'n_neighbors': K}
model = skl_ms.GridSearchCV(estimator=skl_nb.KNeighborsClassifier(), cv=n_fold, param_grid=parameters, scoring='recall_macro', n_jobs=-1)

model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(skl_me.classification_report(y_test, prediction))
print(model.best_params_)
'''