import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import Preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

path = "training_data_vt2025.csv"
df = pd.read_csv(path, na_values="?", dtype={"ID": str}).dropna().reset_index()
df = Preprocess(df)
df_test, df_train = train_test_split(df, test_size=0.2, random_state=42)
x_test = df_test.drop(columns=["increase_stock"])
y_test = df_test["increase_stock"]
x_train = df_train.drop(columns=["increase_stock"])
y_train = df_train["increase_stock"]
rfc = RandomForestClassifier(n_estimators=500, max_depth=50)
model = rfc.fit(x_train, y_train)
y_predict = model.predict(x_test)
con_matrix = confusion_matrix(y_test, y_predict)
ConfusionMatrixDisplay(con_matrix).plot(cmap="winter")
plt.show()
