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
<<<<<<< HEAD:SingleMulti-LayerNN.py

day_info = df['day of week'] # 0=Monday, 1=Tuesday, ..., 6=Sunday

#print(data.head())
=======
#print(data)
>>>>>>> origin/Second:ModularNeuralSystem.py

# prediction model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# global variables
<<<<<<< HEAD:SingleMulti-LayerNN.py
NeuronsNo = 25
OutputNo = 24
EpochsNo = 20

X = df[['L(i-1)', 'L(i-2)', 'L(i-3)', 'L(i-22)', 'L(i-23)', 'L(i-24)', 'L(i-25)', 'L(i-26)', 'mT(tree_hours)', 'mT(previous_day)', 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos', 'hour_sin', 'hour_cos']]
list = []
for i in range(1,25):
    list.append(f'next_load_{i}')
y = df[list]
=======
NeuronsNo = 50 # number of neurons in the hidden layer
OutputNo = 1 # number of neurons in the output layer
EpochsNo = 20 # number of epochs for training

# In/Output data
X = data[['L(i-1)', 'L(i-2)', 'L(i-3)', 'L(i-22)', 'L(i-23)', 'L(i-24)', 'L(i-25)', 'L(i-26)', 'mT(tree_hours)', 'mT(previous_day)', 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos']]
y = df['next_load_1']
>>>>>>> origin/Second:ModularNeuralSystem.py

# Normalization
other_scaler = MinMaxScaler(feature_range=(0, 1))
temp_scaler = MinMaxScaler(feature_range=(-1, 1))

for temp in [f'mT(tree_hours)', f'mT(previous_day)']:
    X[temp] = temp_scaler.fit_transform(X[[temp]])

for col in [f'L(i-{t})' for t in [1, 2, 3, 22, 23, 24, 25, 26]]:
    X[col] = other_scaler.fit_transform(X[[col]])

X = np.array(X)
y = np.array(y)
<<<<<<< HEAD:SingleMulti-LayerNN.py
X_day = np.array(day_info)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, day_train, day_test = train_test_split(X, y, X_day, test_size=0.2)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)), 
    tf.keras.layers.Dense(NeuronsNo, activation='sigmoid'),
    tf.keras.layers.Dense(OutputNo, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['mape'])

trained_model = model.fit(X_train, y_train, epochs=EpochsNo, batch_size=32, validation_split=0.2)
=======
hours = df['hour'].values

# model for every hour
models = {}
Pi_vector = []
mape_vector = []

# indexes of columns L(i-1), L(i-2), L(i-3)
Li1 = 0
Li2 = 1
Li3 = 2

starting_hour = np.where(hours == 5)[0][0] # first hour in the dataset
L_i_1 = X[starting_hour][Li1]
L_i_2 = X[starting_hour][Li2]
L_i_3 = X[starting_hour][Li3]
>>>>>>> origin/Second:ModularNeuralSystem.py

for hour in range(24):
    
    hour_filter = hours == hour
    X_hour = X[hour_filter].copy()
    y_hour = y[hour_filter]

<<<<<<< HEAD:SingleMulti-LayerNN.py
mape_by_day = {}
# Interation over the days of the week
for day in range(7):
    indices = np.where(day_test == day)
    y_true_day = y_test[indices]
    y_pred_day = y_pred[indices]
    
    if len(y_true_day) > 0:
        mape = mean_absolute_percentage_error(y_true_day, y_pred_day)
        mape_by_day[day] = mape * 100 
    else:
        mape_by_day[day] = None


# Visualization of the mape for each day of the week
week_days = {
    0 : 'Monday', 
    1 : 'Tuesday', 
    2 : 'Wednesday',
    3 : 'Thursday', 
    4 : 'Friday',
    5 : 'Saturday',
    6 : 'Sunday'
    }
=======
    X_hour[hour][Li1] = L_i_1
    X_hour[hour][Li2] = L_i_2
    X_hour[hour][Li3] = L_i_3

    X_train, X_test, y_train, y_test = train_test_split(X_hour, y_hour, test_size=0.2)

    models[hour] = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)), 
        tf.keras.layers.Dense(NeuronsNo, activation='sigmoid'),
        tf.keras.layers.Dense(OutputNo, activation='linear')
    ])

    models[hour].compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['mape'])
    models[hour].fit(X_train, y_train, epochs=EpochsNo, batch_size=32, validation_split=0.2, verbose=0)
    y_pred = models[hour].predict(X_test)
    
    Pi_vector.append(f"{y_pred[0][0]:.2f}")
    mape = mean_absolute_percentage_error(y_test, y_pred) 
    MapePercent = mape * 100
    mape_vector.append(f"{MapePercent:.2f}")
    print(f'MAPE: {MapePercent:.2f}%')

    L_i_3 = L_i_2
    L_i_2 = L_i_1
    L_i_1 = float(f"{y_pred[0][0]:.2f}")
>>>>>>> origin/Second:ModularNeuralSystem.py

print()
print(f"{NeuronsNo} neurons in the hidden layer")
print(f"Epochs: {EpochsNo}")
print()
<<<<<<< HEAD:SingleMulti-LayerNN.py
mape_sum = 0
for day, mape in mape_by_day.items():
    day_name = week_days.get(day, f"{day}")
    mape_sum += mape
    print(f"{day_name}: MAPE = {mape:.2f}%" if mape else f"Day {day}: No data")
print()
print(f"Average MAPE: {mape_sum/7:.2f}%")
=======

mape_sum = 0
for i in range(len(Pi_vector)):
    print(f"Load for hour number {i+1}:", Pi_vector[i], "\tMAPE:", mape_vector[i],"%")
    mape_sum += float(mape_vector[i])

avg_mape = mape_sum / len(Pi_vector)
print()
print(f"Average MAPE: {avg_mape:.2f}%")
>>>>>>> origin/Second:ModularNeuralSystem.py
print()
