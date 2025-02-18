import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import Preprocess, Correlation

path = "training_data_vt2025.csv"
df = pd.read_csv(path, na_values="?", dtype={"ID": str})
n_hour = np.zeros(24)
n_day = np.zeros(7)
n_month = np.zeros(12)
n_weekday = np.zeros(2)
n_summertime = np.zeros(2)
n_goodweather = np.zeros(2)

for idx in df.index:
    if df.loc[idx, "increase_stock"] == 1:
        hour = int(df.loc[idx, "hour_of_day"])
        n_hour[hour] += 1

for idx in df.index:
    if df.loc[idx, "increase_stock"] == 1:
        day = int(df.loc[idx, "day_of_week"])
        n_day[day] += 1

for idx in df.index:
    if df.loc[idx, "increase_stock"] == 1:
        month = int(df.loc[idx, "month"]) - 1
        n_month[month] += 1

for idx in df.index:
    if df.loc[idx, "increase_stock"] == 1:
        weekday = int(df.loc[idx, "weekday"])
        n_weekday[weekday] += 1

for idx in df.index:
    if df.loc[idx, "increase_stock"] == 1:
        summertime = int(df.loc[idx, "summertime"])
        n_summertime[summertime] += 1

df = Preprocess(df)
for idx in df.index:
    if df.loc[idx, "increase_stock"] == 1:
        weather = int(df.loc[idx, "good_weather"])
        n_goodweather[weather] += 1

plt.figure(figsize=(10, 6))
plt.plot(range(24), n_hour, "o-")
plt.xlabel("Hour of the Day")
plt.ylabel("Number of High bike demand Events")
plt.title("High bike demand Events by Hour of the Day")
plt.xticks(range(24))
plt.grid(True)
plt.savefig("figHour")

plt.figure(figsize=(10, 6))
plt.plot(range(7), n_day, "o-")
plt.xlabel("day of the week")
plt.ylabel("Number High bike demand Events")
plt.title("High bike demand Events by day of week")
plt.xticks(range(7))
plt.grid(True)
plt.savefig("figDay")

plt.figure(figsize=(10, 6))
plt.plot(range(12), n_month, "o-")
plt.xlabel("month")
plt.ylabel("Number of High bike demand Events")
plt.title("High bike demand Events by month")
plt.xticks(range(12))
plt.grid(True)
plt.savefig("figMonth")

plt.figure(figsize=(10, 6))
plt.bar(range(2), n_weekday)
plt.xlabel("weekday")
plt.ylabel("Number of High bike demand Events")
plt.title("High bike demand Events by weekday")
plt.xticks(range(2))
plt.savefig("figWeekday")

plt.figure(figsize=(10, 6))
plt.bar(range(2), n_summertime)
plt.xlabel("summertime")
plt.ylabel("Number of High bike demand Events")
plt.title("High bike demand Events by summertime")
plt.xticks(range(2))
plt.savefig("figSummertime")

plt.figure(figsize=(10, 6))
plt.bar(range(2), n_goodweather)
plt.xlabel("goodweather")
plt.ylabel("Number of High bike demand Events")
plt.title("High bike demand Events by goodweather")
plt.xticks(range(2))
plt.savefig("figGoodweather")
