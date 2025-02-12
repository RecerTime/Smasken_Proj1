import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.metrics as skl_me

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

std_scalar = skl_pre.StandardScaler()
min_max_scalar = skl_pre.MinMaxScaler()


def Preprocess(df):
    df_length = len(df)
    # Turn hour of day, day of week and month into sin terms to reflect cyclical nature
    df["hour_of_day"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["month"] = np.sin(2 * np.pi * df["month"] / 12)

    # Create workday flag
    df["workday"] = (df["weekday"] == 1) & (df["holiday"] == 0)

    # Scale features correctly using 2D input
    df["dew"] = min_max_scalar.fit_transform(df[["dew"]])
    df["precip"] = min_max_scalar.fit_transform(df[["precip"]])
    df["windspeed"] = std_scalar.fit_transform(df[["windspeed"]])
    df["temp"] = std_scalar.fit_transform(df[["temp"]])
    df["visibility"] = std_scalar.fit_transform(df[["visibility"]])

    return df
