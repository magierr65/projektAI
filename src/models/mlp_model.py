import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.feature_engineering import load_and_preprocess_data
from src.visualization.visualize import plot_mape_by_day, plot_actual_vs_predicted

def run_mlp_training(df):
    model_name = 'mlp'
    NeuronsNo = 25
    OutputNo = 24
    EpochsNo = 20
    X = df[['L(i-1)', 'L(i-2)', 'L(i-3)', 'L(i-22)', 'L(i-23)', 'L(i-24)', 'L(i-25)', 'L(i-26)', 'mT(tree_hours)',
            'mT(previous_day)', 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos', 'hour_sin', 'hour_cos']]
    list_y = [f'next_load_{i}' for i in range(1, 25)]
    y = df[list_y]
    
    other_scaler = MinMaxScaler(feature_range=(0, 1))
    temp_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_normalized = X.copy()
    for temp in ['mT(tree_hours)', 'mT(previous_day)']:
        X_normalized[temp] = temp_scaler.fit_transform(X[[temp]])
    for col in [f'L(i-{t})' for t in [1, 2, 3, 22, 23, 24, 25, 26]]:
        X_normalized[col] = other_scaler.fit_transform(X[[col]])
    
    X = np.array(X_normalized)
    y = np.array(y)
    X_day = np.array(df['day of week'])
    
    X_train, X_test, y_train, y_test, day_train, day_test = train_test_split(X, y, X_day, test_size=0.2, random_state=42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(NeuronsNo, activation='sigmoid'),
        tf.keras.layers.Dense(OutputNo, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), metrics=['mape'])
    model.fit(X_train, y_train, epochs=EpochsNo, batch_size=32, validation_split=0.2, verbose=1)
    
    y_pred = model.predict(X_test)

    mape_by_day = {}
    for day in range(7):
        indices = np.where(day_test == day)
        if len(indices[0]) > 0:
            mape = mean_absolute_percentage_error(y_test[indices], y_pred[indices]) * 100
            mape_by_day[day] = mape
        else:
            mape_by_day[day] = None

    week_days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    print(f"\n{NeuronsNo} neurons in the hidden layer, Epochs: {EpochsNo}\n")
    total_mape = [m for m in mape_by_day.values() if m is not None]
    if total_mape:
        print(f"Overall Average MAPE: {np.mean(total_mape):.2f}%\n")

    # Wizualizacje
    plot_mape_by_day(mape_by_day, model_name, title='MLP: MAPE by Day')
    
    random_index = random.randint(0, len(y_test) - 1)
    plot_actual_vs_predicted(y_test[random_index], y_pred[random_index], model_name,
                             title=f'MLP: Actual vs. Predicted')

if __name__ == '__main__':
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(project_root, 'data', 'combined_data.csv')
    
    df_processed = load_and_preprocess_data(data_path)
    run_mlp_training(df_processed)
