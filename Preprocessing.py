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

scalar = skl_pre.StandardScaler()


def Preprocess(df):
    # Turn hour of day, day of week and month into sin terms to reflect cyclical nature
    df["hour_of_day"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["month"] = np.sin(2 * np.pi * df["month"] / 12)
    # Make a rush hour flag
    # Make a workday flag which is all days where it is weekday and not holiday
    # Normalize all non binary values using either min max or Z dont know which
    return df
