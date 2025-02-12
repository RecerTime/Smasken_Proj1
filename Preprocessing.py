import numpy as np

import sklearn.preprocessing as skl_pre

'''
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
'''

colums_to_remove = ["snow", "cloudcover", "holiday", "precip", "snowdepth"]

def Correlation(df, catagory='increase_stock'):
    # Display correlations between features and target
    feature_correlation = df.corr()[catagory].sort_values(ascending=False)
    print(f'Feature correlations with {catagory}:\n', feature_correlation)
    return feature_correlation

std_scalar = skl_pre.StandardScaler()
min_max_scalar = skl_pre.MinMaxScaler()

def Preprocess(df):
    #df['increase_stock'] = df['increase_stock'].apply(lambda x: 1 if x == 'high_bike_demand' else 0)
    
    # Create workday flag
    #df["workday"] = (df["weekday"] == 1) & (df["holiday"] == 0)
    #df['workday'] = df['workday'].apply(lambda x: 1 if x else 0)

    # Create good weather flag
    df["good_weather"] = (df["precip"] == 0) & (df["snowdepth"] == 0)
    df["good_weather"] = df["good_weather"].apply(lambda x: 1 if x else 0)

    # Create night flag
    df["night"] = df["hour_of_day"].apply(lambda x: 1 if x < 5 or x > 20 else 0)

    # Create rush_hour flag
    #df["rush_hour"] = df["hour_of_day"].apply(lambda x: 1 if x <= 6 and x >= 9 or x >= 16 and x <= 19 else 0)

    # Turn hour of day, day of week and month into sin terms to reflect cyclical nature
    df["hour_of_day"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["month"] = np.sin(2 * np.pi * df["month"] / 12)
        
    # Scale features correctly using 2D input
    df["dew"] = min_max_scalar.fit_transform(df[["dew"]])

    #df["precip"] = min_max_scalar.fit_transform(df[["precip"]])
    #df["windspeed"] = std_scalar.fit_transform(df[["windspeed"]])

    df["temp"] = std_scalar.fit_transform(df[["temp"]])
    df["visibility"] = std_scalar.fit_transform(df[["visibility"]])

    df = df.drop(columns=colums_to_remove)

    print(df)

    return df
