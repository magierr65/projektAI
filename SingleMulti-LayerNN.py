#preprocessing
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r'combined_data.csv')
print(df.head())

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
df['hour'] = df['time'].dt.hour
df['hour_sin'] = np.sin(df['hour'] * 2 * np.pi / 24)
df['hour_cos'] = np.cos(df['hour'] * 2 * np.pi / 24)

for i in [1, 2, 3, 22, 23, 24, 25, 26]:
    df[f'L(i-{i})'] = df['total_load'].shift(i)
    df[f'temp_t-{i}'] = df['temperature'].shift(i)

df['mT(tree_hours)'] = (df[['temp_t-1', 'temp_t-2', 'temp_t-3']].fillna(method='ffill')).mean(axis=1)
df['mT(previous_day)'] = (df[['temp_t-22', 'temp_t-23', 'temp_t-24','temp_t-25','temp_t-26',]].fillna(method='ffill')).mean(axis=1)

for i in range(1,25):
    df[f'next_load_{i}'] = df['total_load'].shift(-i)

df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

columns = ['L(i)', 'L(i-1)', 'L(i-2)', 'L(i-3)', 'L(i-22)', 'L(i-23)', 'L(i-24)', 'L(i-25)', 'L(i-26)', 'mT(tree_hours)', 'mT(previous_day)', 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos', 'hour_sin', 'hour_cos']
data = df[columns] 

day_info = df['day of week'] # 0=Monday, 1=Tuesday, ..., 6=Sunday

#print(data.head())

# prediction model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# global variables
NeuronsNo = 25
OutputNo = 24
EpochsNo = 20

X = df[['L(i-1)', 'L(i-2)', 'L(i-3)', 'L(i-22)', 'L(i-23)', 'L(i-24)', 'L(i-25)', 'L(i-26)', 'mT(tree_hours)', 'mT(previous_day)', 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos', 'hour_sin', 'hour_cos']]
list = []
for i in range(1,25):
    list.append(f'next_load_{i}')
y = df[list]

# Normalization
other_scaler = MinMaxScaler(feature_range=(0, 1))
temp_scaler = MinMaxScaler(feature_range=(-1, 1))

for temp in [f'mT(tree_hours)', f'mT(previous_day)']:
    X[temp] = temp_scaler.fit_transform(X[[temp]])

for col in [f'L(i-{t})' for t in [1, 2, 3, 22, 23, 24, 25, 26]]:
    X[col] = other_scaler.fit_transform(X[[col]])

X = np.array(X)
y = np.array(y)
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

y_pred = model.predict(X_test)

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

print()
print(f"{NeuronsNo} neurons in the hidden layer")
print(f"Epochs: {EpochsNo}")
print()
mape_sum = 0
for day, mape in mape_by_day.items():
    day_name = week_days.get(day, f"{day}")
    mape_sum += mape
    print(f"{day_name}: MAPE = {mape:.2f}%" if mape else f"Day {day}: No data")
print()
print(f"Average MAPE: {mape_sum/7:.2f}%")
print()
