import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from matplotlib.ticker import PercentFormatter
import netCDF4
from netCDF4 import Dataset


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

    # podpunkt 1
    active = confirmed.subtract(deaths).subtract(recovered)
    # print(active)

    # podpunkt 2
    deaths.columns = pd.to_datetime(deaths.columns).month
    deaths = deaths.groupby(deaths.columns, axis=1).sum()

    recovered.columns = pd.to_datetime(recovered.columns).month
    recovered = recovered.groupby(recovered.columns, axis=1).sum()

    death_rate = deaths.divide(recovered)
    # print(death_rate)

    act = pd.pivot_table(active, columns=['Country/Region'], aggfunc=np.sum).sum()

    for country in act.index:  # dopytac o co kaman
        if act[country] <= 100:
            print(country, r[country])

    # Task 2
    zad2(confirmed)


def zad2(confirmed):
    confirmed_seven = confirmed.copy()
    confirmed_factor = confirmed.copy()

    for i in range(6, len(confirmed_seven.columns)):

        last_7 = 0
        for j in range(6):
            last_7 += confirmed[confirmed.columns[i-j]]

        confirmed_seven[confirmed_seven.columns[i]] = last_7/7
        confirmed_factor[confirmed_factor.columns[i]] = confirmed_seven[confirmed_seven.columns[i]] / confirmed_seven[confirmed_seven.columns[i-5]]

    for i in range(6):
        confirmed_seven.drop([confirmed_seven.columns[0]], axis=1, inplace=True)
        confirmed_factor.drop([confirmed_factor.columns[0]], axis=1, inplace=True)

    print(confirmed_seven)
    print(confirmed_factor)

    #weather_data(confirmed_factor)


def weather_data(data):
    weather_max = Dataset('TerraClimate_tmax_2018.nc')
    weather_min = Dataset('TerraClimate_tmin_2018.nc')
    frames = []

    for i in range(12):
        w_max = pd.DataFrame(weather_max['tmax'][i])
        w_min = pd.DataFrame(weather_min['tmin'][i])
        w_mean = (w_max + w_min) / 2
        frames.append(w_mean)
        # w_mean_all = pd.concat([w_mean_all, w_mean])

    #final = pd.concat(frames)
    month = 7  # styczeń

    plt.imshow(weather_max['tmax'][month])
    plt.show()


if __name__ == "__main__":
    zad1()

