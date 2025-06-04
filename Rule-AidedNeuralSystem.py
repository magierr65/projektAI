#preprocessing
import pandas as pd
import numpy as np
import datetime
import holidays
from datetime import datetime, time

# finding extra holidays days
holiday_dates_by_year = {}
for i in range(2020, 2025):
    polish_holidays = holidays.Poland(years=i)
    dates = [date.strftime('%Y-%m-%d') for date, _ in polish_holidays.items()]
    holiday_dates_by_year[i] = dates

# Add extra non-working days
extra_holidays_suffixes = ['12-31', '05-02', '12-24']
for year in holiday_dates_by_year:
    for day in extra_holidays_suffixes:
        holiday_dates_by_year[year].append(f'{year}-{day}')

# Remove selected holidays
to_remove = {
    2020: ['2020-05-31', '2020-01-06'],
    2021: ['2021-05-23', '2021-01-06'],
    2022: ['2022-06-05', '2022-01-06'],
    2023: ['2023-05-28', '2023-01-06'],
    2024: ['2024-05-19', '2024-01-06']
}
for year, dates in to_remove.items():
    for date in dates:
        if date in holiday_dates_by_year[year]:
            holiday_dates_by_year[year].remove(date)

# Flatten all holiday dates into a list
extra_holiday_dates = [pd.to_datetime(date) for year in holiday_dates_by_year for date in holiday_dates_by_year[year]]
extra_holiday_dates = sorted(list(set(extra_holiday_dates)))

# Find untypical days of types 2–6
untypical_days = set()

for date in extra_holiday_dates:
    # Type 2: Monday before extra holiday
    day_before = date - pd.Timedelta(days=1)
    if day_before.weekday() == 0:  # Monday
        untypical_days.add(day_before)

    # Type 3: Tue, Wed, Thu after extra holiday
    for offset in [1, 2, 3]:
        d = date + pd.Timedelta(days=offset)
        if d.weekday() in [1, 2, 3]:  # Tuesday to Thursday
            untypical_days.add(d)

    # Type 4: Friday after extra holiday
    d = date + pd.Timedelta(days=1)
    if d.weekday() == 4:
        untypical_days.add(d)

    # Type 5: Saturday after extra holiday
    d = date + pd.Timedelta(days=1)
    if d.weekday() == 5:
        untypical_days.add(d)

    # Type 6: Saturday two days after extra holiday
    d = date + pd.Timedelta(days=2)
    if d.weekday() == 5:
        untypical_days.add(d)

# Combine all days to remove
all_days_to_remove = set(extra_holiday_dates) | untypical_days
all_days_to_remove = pd.to_datetime(sorted(all_days_to_remove))

# Load the dataset
df = pd.read_csv(r'combined_data.csv')

###############

df_original = df
df_original['time'] = pd.to_datetime(df_original['time'])

date_frame_extra_holidays = df_original[df_original['time'].dt.normalize().isin(extra_holiday_dates)]
date_frame_extra_holidays = date_frame_extra_holidays.sort_values('time').reset_index(drop=True)

date_frame_special_days = df_original[df_original['time'].dt.normalize().isin(all_days_to_remove)]
date_frame_special_days = date_frame_special_days.sort_values('time').reset_index(drop=True)

#date_frame_extra_holidays.to_csv(r"C:\Users\jakub\OneDrive\Pulpit\AGH\semestr4\Sztuczna inteligencja\projekt\projektAI\extra_holidays.csv", index=False)
#date_frame_special_days.to_csv(r"C:\Users\jakub\OneDrive\Pulpit\AGH\semestr4\Sztuczna inteligencja\projekt\projektAI\special_days.csv", index=False)

df = df_original[~df_original['time'].isin(date_frame_special_days['time'])]
df.to_csv(r"C:\Users\jakub\OneDrive\Pulpit\AGH\semestr4\Sztuczna inteligencja\projekt\projektAI\ready.csv", index=False)
################

# Removing rows with extra holidays and untypical days
#df['time'] = pd.to_datetime(df['time'])
#df = df[~df['time'].dt.normalize().isin(all_days_to_remove)]

#print(df.head())

df['L(i)'] = df['total_load']
# Power load at the three previous hours
df['L(i-1)'] = df['total_load'].shift(1)
df['L(i-2)'] = df['total_load'].shift(2)
df['L(i-3)'] = df['total_load'].shift(3)

# Power load on the previous day at the same hour and for neighbouring ones:
df['L(i-22)'] = df['total_load'].shift(22)
df['L(i-23)'] = df['total_load'].shift(23)
df['L(i-24)'] = df['total_load'].shift(24)
df['L(i-25)'] = df['total_load'].shift(25)
df['L(i-26)'] = df['total_load'].shift(26)

# Mean temperature at last three hours
df['mT(tree_hours)'] = ( df['temperature'].shift(1) + df['temperature'].shift(2) + df['temperature'].shift(3) ) / 3

# Mean temperature on the previous day at the same hour and for neighbouring hours
df['mT(previous_day)'] = ( df['temperature'].shift(22) + df['temperature'].shift(23) + df['temperature'].shift(24) + df['temperature'].shift(25) + df['temperature'].shift(26) ) / 5

# The number of the day in week represented on unit vector
df['day of week'] = df['time'].dt.dayofweek # 0=Monday, 1=Tuesday, ..., 6=Sunday
df['weekday_sin'] = np.sin(df['day of week'] * 2 * np.pi / 7)
df['weekday_cos'] = np.cos(df['day of week'] * 2 * np.pi / 7)

# The number of the day in year represented on unit vector
df['day_of_year'] = df['time'].dt.dayofyear.astype(int) # 1-365
df['days_in_year'] = df['time'].dt.is_leap_year.map(lambda x: 366 if x else 365)
df['yearday_sin'] = np.sin(df['day_of_year'] * 2 * np.pi / df['days_in_year'])
df['yearday_cos'] = np.cos(df['day_of_year'] * 2 * np.pi / df['days_in_year'])

# The hour for which the forecasting is done represented on unit vector
df['hour'] = df['time'].dt.hour # 0-23
df['hour_sin'] = np.sin(df['hour'] * 2 * np.pi / 24)
df['hour_cos'] = np.cos(df['hour'] * 2 * np.pi / 24)

for i in range(1,25):
    df[f'next_load_{i}'] = df['total_load'].shift(-i)

# Fill missing values
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

columns = ['hour', 'L(i-1)', 'L(i-2)', 'L(i-3)', 'L(i-22)', 'L(i-23)', 'L(i-24)', 'L(i-25)', 'L(i-26)', 'mT(tree_hours)', 'mT(previous_day)', 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos']
data = df[columns] 

#data.to_csv(r"C:\Users\jakub\OneDrive\Pulpit\AGH\semestr4\Sztuczna inteligencja\projekt\projektAI\preprocessed_data.csv", index=False)

#print(data)


# prediction model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# global variables
NeuronsNo = 50 # number of neurons in the hidden layer
OutputNo = 1 # number of neurons in the output layer
EpochsNo = 100 # number of epochs for training
K = 5 # number od modules
START_HOUR = 0 # hour from which the prediction starts

# In/Output data
X = data[['L(i-1)', 'L(i-2)', 'L(i-3)', 'L(i-22)', 'L(i-23)', 'L(i-24)', 'L(i-25)', 'L(i-26)', 'mT(tree_hours)', 'mT(previous_day)', 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos']]
y = df['next_load_1']

# Normalization
other_scaler = MinMaxScaler(feature_range=(0, 1))
temp_scaler = MinMaxScaler(feature_range=(-1, 1))

for temp in [f'mT(tree_hours)', f'mT(previous_day)']:
    X[temp] = temp_scaler.fit_transform(X[[temp]])

for col in [f'L(i-{t})' for t in [1, 2, 3, 22, 23, 24, 25, 26]]:
    X[col] = other_scaler.fit_transform(X[[col]])

X = np.array(X)
y = np.array(y)
hours = df['hour'].values

# model for every hour
models = {}
Pi_vector = []
mape_vector = []

# indexes of columns L(i-1), L(i-2), L(i-3)
Li1 = 0
Li2 = 1
Li3 = 2

starting_hour = np.where(hours == START_HOUR)[0][0] # first hour in the dataset
L_i_1 = X[starting_hour][Li1]
L_i_2 = X[starting_hour][Li2]
L_i_3 = X[starting_hour][Li3]

pred = {h: [] for h in range(24)}

# iterating from starting hours
ordered_hours = [(START_HOUR + i) % 24 for i in range(24)]

for hour in ordered_hours:
    hour_filter = hours == hour
    X_hour = X[hour_filter].copy()
    y_hour = y[hour_filter]

    # (0) first row of data for a given hour
    X_hour[0][Li1] = L_i_1
    X_hour[0][Li2] = L_i_2
    X_hour[0][Li3] = L_i_3

    X_train, X_test, y_train, y_test = train_test_split(X_hour, y_hour, test_size=0.2)

    for k in range(K):
        models[hour] = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X.shape[1],)), 
            tf.keras.layers.Dense(NeuronsNo, activation='sigmoid'),
            tf.keras.layers.Dense(OutputNo, activation='linear')
            ])

        models[hour].compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['mape'])
        models[hour].fit(X_train, y_train, epochs=EpochsNo, batch_size=32, validation_split=0.2, verbose=0)
        y_pred = models[hour].predict(X_test)
        pred[hour].append(y_pred.flatten())
        
    avg_pred = np.full_like(y_test, fill_value=sum(pred[hour]) / K, dtype=np.float32) # dimension indentical to y_test
    Pi_vector.append(avg_pred[0])

    mape = mean_absolute_percentage_error(y_test, avg_pred)
    MapePercent = mape * 100 / 2
    mape_vector.append(f"{MapePercent:.2f}")
    print(f'HOUR: {hour} --- MAPE: {MapePercent:.2f}%')

    L_i_3 = L_i_2
    L_i_2 = L_i_1
    L_i_1 = float(avg_pred[0])

print()
print(f"Committee Neural System with {K} modules")
print(f"{NeuronsNo} neurons in the hidden layer")
print(f"Epochs: {EpochsNo}")
print()


mape_sum = 0
for i in range(len(Pi_vector)):
    print(f"Load for hour number {(START_HOUR+i)%24}:", Pi_vector[i], "\tMAPE:", mape_vector[i],"%")
    mape_sum += float(mape_vector[i])

avg_mape = mape_sum / len(Pi_vector)
print()
print(f"Average MAPE: {avg_mape:.2f}%")
