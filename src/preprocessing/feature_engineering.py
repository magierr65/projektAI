import pandas as pd
import numpy as np
import holidays

def create_features(df):
    df['L(i)'] = df['total_load']
    df['L(i-1)'] = df['total_load'].shift(1)
    df['L(i-2)'] = df['total_load'].shift(2)
    df['L(i-3)'] = df['total_load'].shift(3)
    df['L(i-22)'] = df['total_load'].shift(22)
    df['L(i-23)'] = df['total_load'].shift(23)
    df['L(i-24)'] = df['total_load'].shift(24)
    df['L(i-25)'] = df['total_load'].shift(25)
    df['L(i-26)'] = df['total_load'].shift(26)
    df['mT(tree_hours)'] = (df['temperature'].shift(1) + df['temperature'].shift(2) + df['temperature'].shift(3)) / 3
    df['mT(previous_day)'] = (df['temperature'].shift(22) + df['temperature'].shift(23) + df['temperature'].shift(24) + df['temperature'].shift(25) + df['temperature'].shift(26)) / 5
    df['time'] = pd.to_datetime(df['time'])
    df['day of week'] = df['time'].dt.dayofweek
    df['weekday_sin'] = np.sin(df['day of week'] * 2 * np.pi / 7)
    df['weekday_cos'] = np.cos(df['day of week'] * 2 * np.pi / 7)
    df['day_of_year'] = df['time'].dt.dayofyear.astype(int)
    df['days_in_year'] = df['time'].dt.is_leap_year.map(lambda x: 366 if x else 365)
    df['yearday_sin'] = np.sin(df['day_of_year'] * 2 * np.pi / df['days_in_year'])
    df['yearday_cos'] = np.cos(df['day_of_year'] * 2 * np.pi / df['days_in_year'])
    df['hour'] = df['time'].dt.hour
    df['hour_sin'] = np.sin(df['hour'] * 2 * np.pi / 24)
    df['hour_cos'] = np.cos(df['hour'] * 2 * np.pi / 24)
    for i in range(1, 25):
        df[f'next_load_{i}'] = df['total_load'].shift(-i)
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    return df

def preprocess_holidays(df):
    holiday_dates_by_year = {}
    for i in range(2020, 2025):
        polish_holidays = holidays.Poland(years=i)
        dates = [date.strftime('%Y-%m-%d') for date, _ in polish_holidays.items()]
        holiday_dates_by_year[i] = dates
    extra_holidays_suffixes = ['12-31', '05-02', '12-24']
    for year in holiday_dates_by_year:
        for day in extra_holidays_suffixes:
            holiday_dates_by_year[year].append(f'{year}-{day}')
    to_remove = {
        2020: ['2020-05-31', '2020-01-06'], 2021: ['2021-05-23', '2021-01-06'],
        2022: ['2022-06-05', '2022-01-06'], 2023: ['2023-05-28', '2023-01-06'],
        2024: ['2024-05-19', '2024-01-06']
    }
    for year, dates in to_remove.items():
        for date in dates:
            if date in holiday_dates_by_year[year]:
                holiday_dates_by_year[year].remove(date)
    extra_holiday_dates = [pd.to_datetime(date) for year in holiday_dates_by_year for date in holiday_dates_by_year[year]]
    extra_holiday_dates = sorted(list(set(extra_holiday_dates)))
    untypical_days = set()
    for date in extra_holiday_dates:
        day_before = date - pd.Timedelta(days=1)
        if day_before.weekday() == 0: untypical_days.add(day_before)
        for offset in [1, 2, 3]:
            d = date + pd.Timedelta(days=offset)
            if d.weekday() in [1, 2, 3]: untypical_days.add(d)
        d = date + pd.Timedelta(days=1)
        if d.weekday() == 4: untypical_days.add(d)
        d = date + pd.Timedelta(days=1)
        if d.weekday() == 5: untypical_days.add(d)
        d = date + pd.Timedelta(days=2)
        if d.weekday() == 5: untypical_days.add(d)
    all_days_to_remove = set(extra_holiday_dates) | untypical_days
    all_days_to_remove = pd.to_datetime(sorted(all_days_to_remove))
    df['time'] = pd.to_datetime(df['time'])
    df = df[~df['time'].dt.normalize().isin(all_days_to_remove)]
    return df

def load_and_preprocess_data(path='data/combined_data.csv'):
    """Wczytuje i przetwarza dane dla standardowych modeli."""
    df = pd.read_csv(path)
    df = create_features(df)
    return df

def load_and_preprocess_rule_aided_data(path='data/combined_data.csv'):
    """Wczytuje i przetwarza dane dla modelu Rule-Aided."""
    df = pd.read_csv(path)
    df = preprocess_holidays(df)
    df = create_features(df)
    return df
