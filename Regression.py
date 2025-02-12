from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis;

#from imblearn.over_sampling import SMOTE

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_data_vt2025.csv')
chosen_model = 'KNN'
bias = 0.5

# Split the DataFrame into inputs (X) and output (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

'''
# Oversample minority class using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)
'''

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

match chosen_model:
    case 'Linear':
        model = LinearRegression(n_jobs=4)
    case 'Logistic':
        model = LogisticRegression(max_iter=10000)
    case 'QDA':
        model = QuadraticDiscriminantAnalysis()
    case 'KNN':
        model = KNeighborsRegressor(n_neighbors=2, n_jobs=4)
    case _:
        print(f'No model with called "{chosen_model}" avalible')

#Creates and trains a LogisticRegression model using the training data
model.fit(X_train, y_train)

#Tests the model using the testing data
y_pred = model.predict(X_test)

#Creates a confusion matrix using the models predictions from the testing data 
cm = confusion_matrix(y_test, (y_pred >= bias).astype(int))

#Calculates and prints the percentage of correct predictions for both high and low demand 
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
print(f"Per-Class Accuracy: Low Demand: {per_class_accuracy[0]:.4f}, High Demand: {per_class_accuracy[1]:.4f}")

#Shows the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.show()