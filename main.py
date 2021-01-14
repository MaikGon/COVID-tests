import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import netCDF4
from netCDF4 import Dataset
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from dateutil.relativedelta import relativedelta
from time import perf_counter


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

    return active, coords


def zad2(active):
    active_seven = active.copy()
    active_factor = active.copy()

    for ctr in active_seven.index:  # Country
        print(ctr)
        for i in range(6, len(active_seven.columns)):
            last_7 = 0
            days = 0
            for j in range(7):
                act = active[active.columns[i - j]][ctr]  # Active case for country for indicated day
                if act >= 100:
                    last_7 += act
                    days += 1

            if days == 0:
                active_seven[active_seven.columns[i]][ctr] = np.nan
                active_factor[active_factor.columns[i]][ctr] = np.nan
            else:
                active_seven[active_seven.columns[i]][ctr] = last_7 / days
                active_factor[active_factor.columns[i]][ctr] = active_seven[active_seven.columns[i]][ctr] / active_seven[active_seven.columns[i - 5]][ctr]

    for i in range(11): # 6+5
        active_factor.drop([active_factor.columns[0]], axis=1, inplace=True)

    active_factor.to_csv("reproduction.csv")

    return active_factor


def weather_data():
    weather_max = Dataset('TerraClimate_tmax_2018.nc')
    weather_min = Dataset('TerraClimate_tmin_2018.nc')
    frames = []

    columns = [(((360 / 8640) * lon) - 180) for lon in range(8640)]
    indexes = [-(((180 / 4320) * lat) - 90) for lat in range(4320)]

    for i in range(12):
        w_max = pd.DataFrame(weather_max['tmax'][i])
        w_min = pd.DataFrame(weather_min['tmin'][i])
        w_mean = (w_max + w_min) / 2
        w_mean.columns = columns
        w_mean.index = indexes
        frames.append(w_mean)
        break # only for faster calcs - delete later

    return frames


def hypothesis(data, frames, coords):
    # Discrete values
    for frame in frames:
        for col in frame.columns:
            frame.loc[frame[col] < 0, col] = 0
            frame.loc[(frame[col] >= 0) & (frame[col] < 10), col] = 1
            frame.loc[(frame[col] >= 10) & (frame[col] < 20), col] = 2
            frame.loc[(frame[col] >= 20) & (frame[col] < 30), col] = 3
            frame.loc[frame[col] >= 30, col] = 4

    # Normalize data
    for country in data.index:
        l_ist = max(data.loc[country].replace(np.inf, 0).fillna(0).tolist())
        data.loc[country] = data.loc[country] / l_ist

    # Add Lat and Long to data
    data = pd.concat([data, coords], axis=1)
    print(data)


if __name__ == "__main__":
    active, coords = zad1()
    # active_factor = zad2(active)

    active_factor = pd.DataFrame(pd.read_csv("reproduction.csv"))
    print(active_factor)
    frames = weather_data()
    hypothesis(active_factor, frames, coords)


