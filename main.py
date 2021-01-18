import pandas as pd
import numpy as np
import netCDF4
from netCDF4 import Dataset
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from dateutil.relativedelta import relativedelta
from time import perf_counter
import math
from scipy.stats import chi2_contingency, normaltest
from statsmodels.stats.power import TTestIndPower
import pycountry_convert as pc


def load_data(data):
    with open(data) as f:
        data = pd.read_csv(f)

    coords = data[['Country/Region', 'Lat', 'Long']]
    coords = coords.groupby(by='Country/Region').mean()

    data.drop(columns=['Province/State', 'Lat', 'Long'], inplace=True)
    data = data.groupby(by='Country/Region').sum()
    # print(data)
    return data, coords


def zad1():
    confirmed, coords = load_data('time_series_covid19_confirmed_global.csv')
    deaths, _ = load_data('time_series_covid19_deaths_global.csv')
    recovered, _ = load_data('time_series_covid19_recovered_global.csv')

    countries_14 = []
    rec = recovered.copy()
    rec = rec.sum(axis=1)

    # Find Countires with no recoveries
    for r in rec.index:
        if rec.loc[r] == 0:
            countries_14.append(r)
            # print(r, rec.loc[r].values[0])

    recovered.columns = pd.to_datetime(recovered.columns)
    confirmed.columns = pd.to_datetime(confirmed.columns)

    for ctr in countries_14:
        for date in confirmed.loc[ctr].index[:-14]:
            val = confirmed.loc[ctr][date]
            date += relativedelta(days=14)
            recovered.loc[ctr][date] += val

    # Find Countires with no death record
    deaths_check = deaths.copy()
    deaths_check = deaths_check.sum(axis=1)

    erase_deaths = []
    for ctr in deaths_check.index:
        if deaths_check.loc[ctr] == 0:
            erase_deaths.append(ctr)
            # print(ctr, deaths_check.loc[ctr])

    # drop countires without death record
    confirmed.drop(erase_deaths, axis=0, inplace=True)
    deaths.drop(erase_deaths, axis=0, inplace=True)
    recovered.drop(erase_deaths, axis=0, inplace=True)
    coords.drop(erase_deaths, axis=0, inplace=True)

    rec_death = recovered.copy()
    rec_death = rec_death.groupby([rec_death.columns.year, rec_death.columns.month], axis=1).sum()

    deaths_r = deaths.copy()
    deaths_r.columns = pd.to_datetime(deaths_r.columns)
    deaths_r = deaths_r.groupby([deaths_r.columns.year, deaths_r.columns.month], axis=1).sum()

    death_rate = deaths_r.divide(rec_death)
    #print('Death_rate: ', death_rate)

    # Active cases
    active = confirmed.subtract(deaths).subtract(recovered)
    active.columns = pd.to_datetime(active.columns)
    #print('Active cases: ', active)

    return active, coords, confirmed, deaths, death_rate


def zad2(active):
    active_seven = active.copy()
    active_factor = active.copy()

    for ctr in active_seven.index:  # Country
        # print(ctr)
        for i in range(6, len(active_seven.columns)):
            last_7 = 0.0
            days = 0.0
            for j in range(7):
                act = active[active.columns[i - j]][ctr]  # Active case for country for indicated day
                if act >= 100:
                    last_7 += act
                    days += 1.0

            if days == 0:
                active_seven[active_seven.columns[i]][ctr] = np.nan
                active_factor[active_factor.columns[i]][ctr] = np.nan
            else:
                active_seven.loc[ctr, active_seven.columns[i]] = last_7 / days
                active_factor.loc[ctr, active_factor.columns[i]] = active_seven[active_seven.columns[i]][ctr] / active_seven[active_seven.columns[i - 5]][ctr]

    for i in range(11): # 6+5
        active_factor.drop([active_factor.columns[0]], axis=1, inplace=True)
    # print(active_factor)
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
        # break # only for faster calcs - delete later

    return frames


def hypothesis(data, frames, coords):
    analysis = TTestIndPower()
    # Dict for further tasks
    data_dict = {
        0: "'< 0'",
        1: "'0 - 10'",
        2: "'10 - 20'",
        3: "'20 - 30'",
        4: "'> 30'"
    }

    # Discrete values
    for frame in frames:
        for col in frame.columns:
            frame.loc[frame[col] < 0, col] = 0
            frame.loc[(frame[col] >= 0) & (frame[col] < 10), col] = 1
            frame.loc[(frame[col] >= 10) & (frame[col] < 20), col] = 2
            frame.loc[(frame[col] >= 20) & (frame[col] < 30), col] = 3
            frame.loc[frame[col] >= 30, col] = 4

    data.columns = pd.to_datetime(data.columns)
    data = data.groupby([data.columns.year, data.columns.month], axis=1).mean()

    # Normalize data
    for country in data.index:
        l_ist = max(data.loc[country].replace(np.inf, 0).fillna(0).tolist())
        data.loc[country] = data.loc[country] / l_ist

    # Add Lat and Long to data
    data = pd.concat([data, coords], axis=1)

    alfa = 0.05
    ctr_found = []
    temp_values = []
    for ctr in data.index:
        # find values from second array by lat and long
        long_closest = np.abs(frames[0].columns - data['Long'][ctr]).argmin()
        long_found = list(frames[0].columns)[long_closest]

        lat_closest = np.abs(frames[0].index - data['Lat'][ctr]).argmin()
        lat_found = list(frames[0].index)[lat_closest]

        for ind, frame in enumerate(frames):
            ctr_found.append(frame.loc[lat_found, long_found])
            year = 2020
            month = ind + 1
            if ind == 0:
                year = 2021
                month = 1
            temp_values.append(data[(year, month)][ctr])

    data_0, data_1, data_2, data_3, data_4 = [], [], [], [], []

    for inde, val in enumerate(ctr_found):
        if val == 0 and not math.isnan(temp_values[inde]):
            data_0.append(temp_values[inde])
        elif val == 1 and not math.isnan(temp_values[inde]):
            data_1.append(temp_values[inde])
        elif val == 2 and not math.isnan(temp_values[inde]):
            data_2.append(temp_values[inde])
        elif val == 3 and not math.isnan(temp_values[inde]):
            data_3.append(temp_values[inde])
        elif val == 4 and not math.isnan(temp_values[inde]):
            data_4.append(temp_values[inde])

    args = []
    names = []
    all_array = [data_0, data_1, data_2, data_3, data_4]  # for further task

    # Add only not empty arrays
    if data_0:
        args.append(data_0)
        names.append('data_0')
    if data_1:
        args.append(data_1)
        names.append('data_1')
    if data_2:
        args.append(data_2)
        names.append('data_2')
    if data_3:
        args.append(data_3)
        names.append('data_3')
    if data_4:
        args.append(data_4)
        names.append('data_4')

    # If there are minimum 2 datasets
    if len(args) >= 2:
        d = np.concatenate([*args])
        k2, p = normaltest(d)

        print('p = ', p)
        if p < alfa:
            print('Możemy odrzucić hipotezę o nie normalności rozkładów')
            print('Wariancje:')
            for arg in args:
                print(np.std(arg))

            f_value, p_value = f_oneway(*args)
            print(f'F-stat: {f_value}, p-val: {p_value}', '\n')

            if p_value < alfa:
                tukey = pairwise_tukeyhsd(np.concatenate([*args]), np.concatenate(
                    [[names[ind]] * len(f) for ind, f in enumerate(args)]))
                print(tukey, '\n')
                dt_frame = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

                for idx in dt_frame.index:
                    # If True in column 'reject' -> significant difference
                    if dt_frame['reject'][idx]:
                        print('Istotna różnica w zbiorze: ', str(dt_frame['group1'][idx]), str(dt_frame['group2'][idx]))
                        d1 = int(str(dt_frame['group1'][idx])[-1])
                        d2 = int(str(dt_frame['group2'][idx])[-1])

                        effect = (np.mean(all_array[int(d1)]) - np.mean(all_array[int(d2)])) / ((np.std(all_array[int(d1)]) + np.std(all_array[int(d2)])) / 2)
                        result = analysis.solve_power(effect, power=None, nobs1=len(all_array[int(d1)]), ratio=len(all_array[int(d2)])/len(all_array[int(d1)]), alpha=alfa)
                        print("Moc zbioru (prawdopodobieństwo niepopełnienia błędu) ", result)
                        print('Pomiędzy przedziałami temperatur', data_dict[d1], ' oraz ', data_dict[d2],
                              'zauważono istotne różnice. Możemy stwierdzić z prawdopodobieństwem '
                              , str(result), ', że temperatura wpływa na szybkość rozprzestrzeniania się wirusa.', '\n')


def hypothesis_part_2(confirmed, deaths, death_rate):
    analysis = TTestIndPower()
    not_europe = []
    for ctr in confirmed.index:
        try:
            country_code = pc.country_name_to_country_alpha2(ctr, cn_name_format="default")
            continent_name = pc.country_alpha2_to_continent_code(country_code)
            if continent_name != 'EU':
                not_europe.append(ctr)
        except:
            if ctr != 'Kosovo':
                not_europe.append(ctr)

    confirmed.drop(not_europe, axis=0, inplace=True)
    deaths.drop(not_europe, axis=0, inplace=True)
    death_rate.drop(not_europe, axis=0, inplace=True)

    deaths_chi = deaths.copy()
    deaths_chi = deaths_chi.sum(axis=1)
    confirmed = confirmed.sum(axis=1)

    alfa = 0.05
    obs = []

    for ctr in confirmed.index:
        conf = confirmed[ctr]
        deat = deaths_chi[ctr]
        obs.append([conf, deat])

    chi2, p, dof, ex = chi2_contingency(obs)
    # print('chi2: ', chi2, '\n', 'p: ', p, '\n', 'dof: ', dof, '\n', 'ex: ', ex)
    print('\n', 'p = ', p)
    if p < alfa:
        print('Wartość p = ', p, ',a więc możemy stwierdzić, że są istotne róznice w śmiertelności pomiędzy krajami w Europie')
    else:
        print('Wartość p = ', p, ',a więc nie możemy stwierdzić, czy są istotne róznice w śmiertelności pomiędzy krajami w Europie')

    args = []
    names = []

    death_rate = death_rate.replace(np.inf, np.nan)

    for ctr in death_rate.index:
        deaths = death_rate.loc[ctr, :].dropna().tolist()
        args.append(deaths)
        names.append(ctr)

    d = np.concatenate([*args])
    k2, p = normaltest(d)

    print('p = ', p)
    if p < alfa:
        print('Możemy odrzucić hipotezę o nie normalności rozkładów')
        print('Wariancje:')
        for arg in args:
            print(np.std(arg))

        f_value, p_value = f_oneway(*args)
        print(f'F-stat: {f_value}, p-val: {p_value}')
        if p_value < alfa:
            tukey = pairwise_tukeyhsd(np.concatenate([*args]), np.concatenate(
                [[names[ind]] * len(f) for ind, f in enumerate(args)]))
            print('\n', tukey, '\n')
            dt_frame = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

            for idx in dt_frame.index:
                # If True in column 'reject' -> significant difference

                if dt_frame['reject'][idx]:
                    print('Istotna różnica w krajach: ', str(dt_frame['group1'][idx]), str(dt_frame['group2'][idx]))
                    d1 = str(dt_frame['group1'][idx])
                    d2 = str(dt_frame['group2'][idx])
                    idx1 = names.index(d1)
                    idx2 = names.index(d2)

                    effect = (np.mean(args[idx1]) - np.mean(args[idx2])) / (
                                (np.std(args[idx1]) + np.std(args[idx2])) / 2)
                    result = analysis.solve_power(effect, power=None, nobs1=len(args[idx1]),
                                                  ratio=len(args[idx2]) / len(args[idx1]), alpha=alfa)
                    print("Moc zbioru (prawdopodobieństwo niepopełnienia błędu) ", result, '\n')
                    print('Pomiędzy krajami', str(dt_frame['group1'][idx]), ' oraz ', str(dt_frame['group2'][idx]),
                          'zauważono istotne różnice. Możemy stwierdzić z prawdopodobieństwem '
                          , str(result), ', że między poszczególnymi krajami w Europie istnieją istotne różnice w śmiertelności.', '\n')


if __name__ == "__main__":
    start = perf_counter()
    active, coords, confirmed, deaths_r, death_rate = zad1()
    # active_factor = zad2(active)

    active_factor = pd.DataFrame(pd.read_csv("reproduction.csv", index_col='Country/Region'))
    frames = weather_data()
    hypothesis(active_factor, frames, coords)
    hypothesis_part_2(confirmed, deaths_r, death_rate)
    stop = perf_counter()
    print('Elapsed time: ', str(stop - start))

