import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import Preprocess, Correlation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

r = 0.45
path = "training_data_vt2025.csv"
df = pd.read_csv(path, na_values="?", dtype={"ID": str})
df = Preprocess(df)
print(Correlation(df))
df_test, df_train = train_test_split(df, test_size=0.2, random_state=42)
x_test = df_test.drop(columns=["increase_stock"])
y_test = df_test["increase_stock"]
x_train = df_train.drop(columns=["increase_stock"])
y_train = df_train["increase_stock"]
rfc = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
)
model = rfc.fit(x_train, y_train)
y_prob = model.predict_proba(x_test)[:, 1]
print(y_prob)
y_predict = (y_prob >= r).astype(int)
con_matrix = confusion_matrix(y_test, y_predict)
ConfusionMatrixDisplay(con_matrix).plot(cmap="winter")
plt.show()
