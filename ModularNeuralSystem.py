#preprocessing
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r'combined_data.csv')

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

# Changing time into datetime format
df['time'] = pd.to_datetime(df['time'])

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
print(data)


# prediction model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# global variables
NeuronsNo = 25
OutputNo = 1

# In/Output data
X = df[['L(i-1)', 'L(i-2)', 'L(i-3)', 'L(i-22)', 'L(i-23)', 'L(i-24)', 'L(i-25)', 'L(i-26)', 'mT(tree_hours)', 'mT(previous_day)', 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos']]
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
#print(y)
x_hour = np.array(df['hour'])

# Split the data into training and testing sets

# model for every hour
models = {}
Pi_vector = []
mape_vector = []

starting_hour = 0
hour_train = ( data['hour'] == starting_hour )
X_hour = X[hour_train]
y_hour = y[hour_train]

Li1 = 1
Li2 = 2
Li3 = 3

L_i_1 = X_hour[starting_hour][Li1]
L_i_2 = X_hour[starting_hour][Li2]
L_i_3 = X_hour[starting_hour][Li3]
temp = []
y_t = []
for hour in range(24):

    X_hour[hour][Li1] = L_i_1
    X_hour[hour][Li2] = L_i_2
    X_hour[hour][Li3] = L_i_3

    # Filter the data for the current hour

    X_train, X_test, y_train, y_test = train_test_split(X_hour, y_hour, test_size=0.2)
    
    models[hour] = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)), 
        tf.keras.layers.Dense(NeuronsNo, activation='sigmoid'),
        tf.keras.layers.Dense(OutputNo, activation='linear')
    ])
    models[hour].compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['mape'])
    models[hour].fit(X_hour, y_hour, epochs=20, batch_size=32, validation_split=0.2)
    y_pred = models[hour].predict(X_test)
    
    temp.append(y_pred)
    y_t.append(y_test)

    Pi_vector.append(f"{y_pred[0][0]:.2f}")
    mape = mean_absolute_percentage_error(y_test, y_pred) 
    MapePercent = mape * 100
    mape_vector.append(f"{MapePercent:.2f}")
    print(f'MAPE: {MapePercent:.2f}%')

    L_i_3 = L_i_2
    L_i_2 = L_i_1
    L_i_1 = float(f"{y_pred[0][0]:.2f}")


print(Pi_vector)
print(mape_vector)
#print(temp)

import matplotlib.pyplot as plt

    
plt.plot(y_t[0], label='Real Load')
plt.legend()
plt.show()
