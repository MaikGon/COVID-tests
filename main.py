import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import netCDF4
from netCDF4 import Dataset
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from datetime import datetime
from dateutil.relativedelta import relativedelta


def load_data(data):
    with open(data) as f:
        data = pd.read_csv(f)

    coords = data[['Country/Region', 'Lat', 'Long']]
    coords = coords.groupby(by='Country/Region').mean()

    data.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
    data = data.groupby(by='Country/Region').sum()
    # print(data.head(5))
    return data, coords


def zad1():
    confirmed, coords = load_data('time_series_covid19_confirmed_global.csv')
    deaths, _ = load_data('time_series_covid19_deaths_global.csv')
    recovered, _ = load_data('time_series_covid19_recovered_global.csv')

    # Find countries with no public recoveries record
    r = pd.pivot_table(recovered, columns=['Country/Region'], aggfunc=np.sum).sum()
    for country in r.index:
        if r[country] == 0:  # or NaN - dopytac
            pass
            # print(country, r[country])

    # Drop countries with no public deaths record
    d = pd.pivot_table(deaths, columns=['Country/Region'], aggfunc=np.sum).sum()
    for country in d.index:
        if d[country] == 0:  # or NaN - dopytac
            pass
            # confirmed.drop(index=country, inplace=True)
            # deaths.drop(index=country, inplace=True)
            # recovered.drop(index=country, inplace=True)

    # subpoint 1
    active = confirmed.subtract(deaths).subtract(recovered)
    # print('Active: ', active)

    # subpoint 3
    countries_14 = []
    rec = recovered.copy()
    rec.columns = pd.to_datetime(rec.columns).year
    rec = rec.groupby(rec.columns, axis=1).sum()

    # Find Countires with no recoveries
    for r in rec.index:
        if rec.loc[r].values == 0:
            countries_14.append(r)
            # print(r, rec.loc[r].values[0])

    recovered.columns = pd.to_datetime(recovered.columns)
    confirmed.columns = pd.to_datetime(confirmed.columns)
    active.columns = pd.to_datetime(active.columns)

    for ctr in countries_14:
        for date in confirmed.loc[ctr].index[:-14]:
            val = confirmed.loc[ctr][date]
            date += relativedelta(days=14)
            recovered.loc[ctr][date] += val

    # subpoint 2
    recovered.columns = pd.to_datetime(recovered.columns).month
    recovered = recovered.groupby(recovered.columns, axis=1).sum()

    deaths_r = deaths.copy()
    deaths_r.columns = pd.to_datetime(deaths_r.columns).month
    deaths_r = deaths_r.groupby(deaths_r.columns, axis=1).sum()

    death_rate = deaths_r.divide(recovered)
    # print('Death_rate: ', death_rate)

    # subpoint 4
    deaths_check = deaths.copy()
    deaths_check.columns = pd.to_datetime(deaths_check.columns).year
    deaths_check = deaths_check.groupby(deaths_check.columns, axis=1).sum()

    erase_deaths = []
    for ctr in deaths_check.index:
        if deaths_check.loc[ctr].values == 0:
            erase_deaths.append(ctr)
            # print(ctr, deaths_check.loc[ctr].values[0])

    # drop countires without death record
    confirmed.drop(erase_deaths, axis=0, inplace=True)
    active.drop(erase_deaths, axis=0, inplace=True)

    # subpoint 5 in task 2

    # Task 2
    zad2(confirmed, active, coords)


def zad2(confirmed, active, coords):
    confirmed_seven = confirmed.copy()
    confirmed_factor = confirmed.copy()

    for ctr in confirmed_seven.index:  # Country
        for i in range(6, len(confirmed_seven.columns)):
            last_7 = 0
            days = 0
            for j in range(7):
                act = active[active.columns[i - j]][ctr]  # Active case for country for indicated day
                if act >= 100:
                    last_7 += confirmed[confirmed.columns[i - j]][ctr]
                    days += 1

            if days == 0:
                confirmed_seven[confirmed_seven.columns[i]][ctr] = np.nan
                confirmed_factor[confirmed_factor.columns[i]][ctr] = np.nan
            else:
                confirmed_seven[confirmed_seven.columns[i]][ctr] = last_7 / days
                confirmed_factor[confirmed_factor.columns[i]][ctr] = confirmed_seven[confirmed_seven.columns[i]][ctr] / confirmed_seven[confirmed_seven.columns[i - 5]][ctr]

    for i in range(11): # 6+5
        confirmed_seven.drop([confirmed_seven.columns[0]], axis=1, inplace=True)
        confirmed_factor.drop([confirmed_factor.columns[0]], axis=1, inplace=True)

    confirmed_seven.to_csv("mean_seven.csv")
    confirmed_factor.to_csv("reproduction.csv")

    #weather_data(confirmed_factor, coords)
    #hypothesis(confirmed_factor, 'x', coords)


def weather_data(data, coords):
    weather_max = Dataset('TerraClimate_tmax_2018.nc')
    weather_min = Dataset('TerraClimate_tmin_2018.nc')
    frames = []

    for i in range(12):
        w_max = pd.DataFrame(weather_max['tmax'][i])
        w_min = pd.DataFrame(weather_min['tmin'][i])
        w_mean = (w_max + w_min) / 2
        columns = [(i - 180) / 24 for i in w_mean.columns]
        indexes = [(i - 90) / 24 for i in w_mean.index]
        w_mean.columns = columns
        w_mean.index = indexes
        frames.append(w_mean)

    plt.imshow(frames[0])
    plt.show()

    hypothesis(data, frames, coords)


def hypothesis(data, frames, coords):

    for country in data.index:
        # normalize data
        l_ist = max(data.loc[country].replace(np.inf, 0).fillna(0).tolist())
        data.loc[country] = data.loc[country] / l_ist

    # Add Lat and Long to data
    data = pd.concat([data, coords], axis=1)
    print(data)


if __name__ == "__main__":
    zad1()

