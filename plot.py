import pandas
import numpy as np
from matplotlib import pyplot as plt

df = pandas.read_csv('training_data_vt2025.csv').groupby('increase_stock')

fig, ax = plt.subplots(3, 2)
fig.tight_layout()
fig.suptitle('training_data_vt2025')

ax[0][0].set_title('hour_of_day')
h_df = df['hour_of_day'].value_counts()
ax[0][0].plot(h_df['high_bike_demand'].sort_index(), label='high_bike_demand')
ax[0][0].plot(h_df['low_bike_demand'].sort_index(), label='low_bike_demand')

ax[1][0].set_title('day_of_week')
d_df = df['day_of_week'].value_counts()
ax[1][0].plot(d_df['high_bike_demand'].sort_index(), label='high_bike_demand')
ax[1][0].plot(d_df['low_bike_demand'].sort_index(), label='low_bike_demand')

ax[2][0].set_title('month')
m_df = df['month'].value_counts()
ax[2][0].plot(m_df['high_bike_demand'].sort_index(), label='high_bike_demand')
ax[2][0].plot(m_df['low_bike_demand'].sort_index(), label='low_bike_demand')

ax[0][1].set_title('holiday')
hd_df = df['holiday'].value_counts()
ax[0][1].bar('high_bike_demand, Holiday', hd_df['high_bike_demand'][1])
ax[0][1].bar('high_bike_demand, non-Holiday', hd_df['high_bike_demand'][0])
ax[0][1].bar('low_bike_demand, Holiday', hd_df['low_bike_demand'][1])
ax[0][1].bar('low_bike_demand, non-Holiday', hd_df['low_bike_demand'][0])

ax[1][1].set_title('weekday')
wd_df = df['weekday'].value_counts()
ax[1][1].bar('high_bike_demand, weekday', wd_df['high_bike_demand'][1])
ax[1][1].bar('high_bike_demand, non-weekday', wd_df['high_bike_demand'][0])
ax[1][1].bar('low_bike_demand, weekday', wd_df['low_bike_demand'][1])
ax[1][1].bar('low_bike_demand, non-weekday', wd_df['low_bike_demand'][0])

ax[2][1].set_title('summertime')
st_df = df['summertime'].value_counts()
ax[2][1].bar('high_bike_demand, summertime', st_df['high_bike_demand'][1])
ax[2][1].bar('high_bike_demand, non-summertime', st_df['high_bike_demand'][0])
ax[2][1].bar('low_bike_demand, summertime', st_df['low_bike_demand'][1])
ax[2][1].bar('low_bike_demand, non-summertime', st_df['low_bike_demand'][0])

for a in ax.flatten():
    a.legend(loc='best')

plt.show()
