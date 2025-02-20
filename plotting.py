import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Preprocessing import Preprocess, Correlation

path = "training_data_vt2025.csv"
data = pd.read_csv(path, na_values="?", dtype={"ID": str})

# Initialize arrays for counts and totals
n_hour, total_hour = np.zeros(24), np.zeros(24)
n_day, total_day = np.zeros(7), np.zeros(7)
n_month, total_month = np.zeros(12), np.zeros(12)
n_weekday, total_weekday = np.zeros(2), np.zeros(2)
n_summertime, total_summertime = np.zeros(2), np.zeros(2)
n_holiday, total_holiday = np.zeros(2), np.zeros(2)
# Calculate numerators and denominators for each category
for idx in data.index:
    # Hour of day
    hour = int(data.loc[idx, "hour_of_day"])
    total_hour[hour] += 1
    if data.loc[idx, "increase_stock"] == 1:
        n_hour[hour] += 1

    # Day of week
    day = int(data.loc[idx, "day_of_week"])
    total_day[day] += 1
    if data.loc[idx, "increase_stock"] == 1:
        n_day[day] += 1

    # Month (adjusted to 0-based index)
    month = int(data.loc[idx, "month"]) - 1
    total_month[month] += 1
    if data.loc[idx, "increase_stock"] == 1:
        n_month[month] += 1

    # Weekday
    weekday = int(data.loc[idx, "weekday"])
    total_weekday[weekday] += 1
    if data.loc[idx, "increase_stock"] == 1:
        n_weekday[weekday] += 1

    # Summertime
    summertime = int(data.loc[idx, "summertime"])
    total_summertime[summertime] += 1
    if data.loc[idx, "increase_stock"] == 1:
        n_summertime[summertime] += 1
    # holiday
    holiday = int(data.loc[idx, "holiday"])
    total_holiday[holiday] += 1
    if data.loc[idx, "increase_stock"] == 1:
        n_holiday[holiday] += 1

# Compute probabilities
prob_hour = n_hour / total_hour
prob_day = n_day / total_day
prob_month = n_month / total_month
prob_weekday = n_weekday / total_weekday
prob_summertime = n_summertime / total_summertime
prob_holiday = n_holiday / total_holiday


data = Preprocess(data)

# Calculate good_weather probabilities
n_goodweather, total_goodweather = np.zeros(2), np.zeros(2)
for idx in data.index:
    weather = int(data.loc[idx, "good_weather"])
    total_goodweather[weather] += 1
    if data.loc[idx, "increase_stock"] == 1:
        n_goodweather[weather] += 1

prob_goodweather = n_goodweather / total_goodweather


def plot_probability(x, prob, xlabel, title, file, bar=False, xticks=None, labels=None):
    plt.figure(figsize=(10, 6))
    if bar:
        plt.bar(range(len(prob)), prob)
        if labels:
            plt.xticks(range(len(prob)), labels)
    else:
        plt.plot(x, prob, "o-")
        plt.xticks(x)
        plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.title(title)
    plt.savefig(file, bbox_inches="tight")
    plt.close()


plot_probability(
    range(24),
    prob_hour,
    "Hour of the Day",
    "Probability of High Bike Demand by Hour",
    "figHour.pdf",
)
plot_probability(
    range(7), prob_day, "Day of the Week", "Probability by Day of Week", "figDay.pdf"
)
plot_probability(range(12), prob_month, "Month", "Probability by Month", "figMonth.pdf")
plot_probability(
    range(2),
    prob_weekday,
    "Weekday",
    "Probability by Weekday",
    "figWeekday.pdf",
    bar=True,
    labels=["Non-Weekday", "Weekday"],
)
plot_probability(
    range(2),
    prob_summertime,
    "Summertime",
    "Probability by Summertime",
    "figSummertime.pdf",
    bar=True,
    labels=["No", "Yes"],
)
plot_probability(
    range(2),
    prob_holiday,
    "Holiday",
    "Probability by holiday",
    "figHoliday.pdf",
    bar=True,
    labels=["No", "Yes"],
)
plot_probability(
    range(2),
    prob_goodweather,
    "Good Weather",
    "Probability by Good Weather",
    "figGoodweather.pdf",
    bar=True,
    labels=["No", "Yes"],
)
