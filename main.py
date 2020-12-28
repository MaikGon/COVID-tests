import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from matplotlib.ticker import PercentFormatter
import os
import json
import datetime


def load_data(data):
    with open(data) as f:
        data = pd.read_csv(f)

    data.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
    data = data.groupby(by='Country/Region').sum()
    # print(data.head(5))
    return data


def zad1():
    confirmed = load_data('time_series_covid19_confirmed_global.csv')
    deaths = load_data('time_series_covid19_deaths_global.csv')
    recovered = load_data('time_series_covid19_recovered_global.csv')

    # podpunkt 1
    active = confirmed.subtract(deaths).subtract(recovered)
    print(active)

    # podpunkt 2
    deaths.columns = pd.to_datetime(deaths.columns).month
    deaths = deaths.groupby(deaths.columns, axis=1).sum()

    recovered.columns = pd.to_datetime(recovered.columns).month
    recovered = recovered.groupby(recovered.columns, axis=1).sum()

    death_rate = deaths.divide(recovered)
    print(death_rate)


if __name__ == "__main__":
    zad1()

