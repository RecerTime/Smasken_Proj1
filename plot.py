import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


path = "training_data_vt2025.csv"
df = pd.read_csv(path, na_values="?", dtype={"ID": str})

n_hour = np.zeros(24)
n_day = np.zeros(7)
n_month = np.zeros(12)

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

plt.figure(figsize=(10, 6))
plt.plot(range(24), n_hour, "o-")
plt.xlabel("Hour of the Day")
plt.ylabel('Number of "increase_stock" Events')
plt.title("Stock Increase Events by Hour of the Day")
plt.xticks(range(24))
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(range(7), n_day, "o-")
plt.xlabel("day of the week")
plt.ylabel('Number of "increase_stock" Events')
plt.title("Stock Increase Events by day of week")
plt.xticks(range(7))
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(range(12), n_month, "o-")
plt.xlabel("month")
plt.ylabel('Number of "increase_stock" Events')
plt.title("Stock Increase Events by month")
plt.xticks(range(12))
plt.grid(True)
plt.show()
