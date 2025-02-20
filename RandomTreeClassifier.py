import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import Preprocess, Correlation
from scipy.stats import randint
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


r = 0.18
path = "training_data_vt2025.csv"
df = pd.read_csv(path, na_values="?", dtype={"ID": str})
df = Preprocess(df)
print(Correlation(df))
x = df.drop(columns="increase_stock")
y = df["increase_stock"].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
rfc = RandomForestClassifier(
    n_estimators=367,
    max_depth=10,
    min_samples_leaf=7,
    min_samples_split=8,
    max_features="log2",
    class_weight="balanced",
    random_state=42,
    bootstrap=False,
)
model = rfc.fit(x_train, y_train)
y_prob = model.predict_proba(x_test)[:, 1]
y_predict = (y_prob >= r).astype(int)
score = classification_report(y_test, y_predict)
print(score)
con_matrix = confusion_matrix(y_test, y_predict)
ConfusionMatrixDisplay(con_matrix).plot(cmap="winter")
plt.savefig("confusionMatrixTree.pdf", bbox_inches="tight")
plt.show()


def hyper_para_optimisation(model):
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [None] + list(range(5, 50, 5)),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced", "balanced_subsample"],
        "bootstrap": [True, False],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=500,
        cv=cv,
        scoring="recall_macro",  #  for better false negative score
        n_jobs=-1,
        random_state=42,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_
