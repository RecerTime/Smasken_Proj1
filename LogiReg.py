import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb

path = "training_data_vt2025.csv"
data = pd.read_csv(path, na_values="?", dtype={"ID": str}).dropna().reset_index()

n = data.shape[0]
numbers = np.arange(n)
numbers_list = numbers.tolist()
random_numbers = random.sample(numbers_list, int((n + 1) / 2))
remaning_random_numbers = list(set(numbers_list) - set(random_numbers))

training_data = data.iloc[random_numbers]
test_data = data.iloc[remaning_random_numbers]
print(training_data)

input = training_data[
    [
        "hour_of_day",
        "day_of_week",
        "month",
        "holiday",
        "weekday",
        "summertime",
        "temp",
        "dew",  # remove
        "humidity",  # remove
        "precip",
        "snow",  # remove
        "snowdepth",
        "windspeed",
        "cloudcover",  # remove
        "visibility",  # remove
    ]
]
output = training_data["increase_stock"]
model = skl_lm.LogisticRegression(max_iter=10000)
model.fit(input, output)
