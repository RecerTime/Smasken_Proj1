from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from Preprocessing import Preprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_raw = pd.read_csv('training_data_vt2025.csv')
df = Preprocess(df_raw)

# Split the DataFrame into inputs (X) and output (y)
X = df.drop(columns='increase_stock')
y = df['increase_stock'].values

# Define hyperparameter grid
param_grid = {'reg_param': np.linspace(0, 1)}

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform GridSearchCV with 10-fold cross-validation on QDA model to get best hyperparameters
model = GridSearchCV(estimator=QuadraticDiscriminantAnalysis(), param_grid=param_grid, cv=50, scoring='accuracy', n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

# Print best hyperparameter
print(f"Best hyperparameters: {model.best_params_}")

#Tests the model using the testing data
y_pred = model.predict(X_test)

#Creates a confusion matrix using the models predictions from the testing data 
cm = confusion_matrix(y_test, y_pred)

# Print classification metrics
print(classification_report(y_test, y_pred))

#Shows the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.show()