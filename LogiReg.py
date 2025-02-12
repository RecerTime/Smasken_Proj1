import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from Preprocessing import Preprocess

import seaborn as sb
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.metrics as skl_me

path = "training_data_vt2025.csv"
data = pd.read_csv(path, na_values="?", dtype={"ID": str}).dropna().reset_index()
new_data = Preprocess(data)
n = data.shape[0]
# Use the sklearn thing instead
numbers = np.arange(n)
numbers_list = numbers.tolist()
random_numbers = random.sample(numbers_list, int((n + 1) / 2))
remaning_random_numbers = list(set(numbers_list) - set(random_numbers))

training_data = data.iloc[random_numbers]
test_data = data.iloc[remaning_random_numbers]
print(training_data)
parameters = [
    "hour_of_day",
    "day_of_week",
    "month",
    "holiday",
    "weekday",
    "summertime",
    "temp",
    "dew",
    "humidity",
    "precip",
    "snow",
    "snowdepth",
    "windspeed",
    "cloudcover",
    "visibility",
]
input = training_data[parameters]
output = training_data["increase_stock"]
model = skl_lm.LogisticRegression(max_iter=10000)
model.fit(input, output)
test_input = test_data[parameters]
test_output = test_data["increase_stock"]
predictions = model.predict(test_input)
con_matrix = skl_me.confusion_matrix(test_output, predictions)
skl_me.ConfusionMatrixDisplay(con_matrix).plot()
plt.show()
