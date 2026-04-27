import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.feature_engineering import load_and_preprocess_data
from src.visualization.visualize import plot_hourly_mape, plot_hourly_predictions_comparison

def run_committee_training(df):
    model_name = 'committee'
    NeuronsNo = 50
    OutputNo = 1
    EpochsNo = 20
    START_HOUR = 5

    X = df[['L(i-1)', 'L(i-2)', 'L(i-3)', 'L(i-22)', 'L(i-23)', 'L(i-24)', 'L(i-25)', 'L(i-26)', 'mT(tree_hours)',
              'mT(previous_day)', 'weekday_sin', 'weekday_cos', 'yearday_sin', 'yearday_cos']]
    y = df['next_load_1']
    
    other_scaler = MinMaxScaler(feature_range=(0, 1))
    temp_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_normalized = X.copy()
    for temp in ['mT(tree_hours)', 'mT(previous_day)']:
        X_normalized[temp] = temp_scaler.fit_transform(X[[temp]])
    for col in [f'L(i-{t})' for t in [1, 2, 3, 22, 23, 24, 25, 26]]:
        X_normalized[col] = other_scaler.fit_transform(X[[col]])
    
    X = np.array(X_normalized)
    y = np.array(y)
    hours = df['hour'].values
    
    Pi_vector = []
    mape_vector = []
    
    Li1, Li2, Li3 = 0, 1, 2
    starting_hour_index = np.where(hours == START_HOUR)[0][0]
    L_i_1, L_i_2, L_i_3 = X[starting_hour_index][Li1], X[starting_hour_index][Li2], X[starting_hour_index][Li3]

    # Get the true 24-hour load values starting from the forecast point
    y_true_24h = df.iloc[starting_hour_index][[f'next_load_{i}' for i in range(1, 25)]].values

    for hour in range(24):
        hour_filter = hours == hour
        X_hour, y_hour = X[hour_filter].copy(), y[hour_filter]
        
        if len(X_hour) > hour:
            X_hour[hour][Li1], X_hour[hour][Li2], X_hour[hour][Li3] = L_i_1, L_i_2, L_i_3

        X_train, X_test, y_train, y_test = train_test_split(X_hour, y_hour, test_size=0.2, random_state=42)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_hour.shape[1],)),
            tf.keras.layers.Dense(NeuronsNo, activation='sigmoid'),
            tf.keras.layers.Dense(OutputNo, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['mape'])
        model.fit(X_train, y_train, epochs=EpochsNo, batch_size=32, validation_split=0.2, verbose=0)
        
        y_pred = model.predict(X_test)
        
        if len(y_pred) > 0:
            predicted_val = y_pred[0][0]
            Pi_vector.append(f"{predicted_val:.2f}")
            L_i_3, L_i_2, L_i_1 = L_i_2, L_i_1, predicted_val
        else:
            Pi_vector.append("0.00")

        mape = mean_absolute_percentage_error(y_test, y_pred) * 100 if len(y_test) > 0 else 0
        mape_vector.append(f"{mape:.2f}")
        print(f'Hour {hour}: MAPE = {mape:.2f}%')

    print(f"\n{NeuronsNo} neurons, Epochs: {EpochsNo}\n")
    avg_mape = np.mean([float(m) for m in mape_vector])
    print(f"Average MAPE: {avg_mape:.2f}%\n")

    plot_hourly_mape(mape_vector, model_name, title='Committee: Hourly MAPE')
    plot_hourly_predictions_comparison(y_true_24h, Pi_vector, model_name, title='Committee: Actual vs. Predicted Load')

if __name__ == '__main__':
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    data_path = os.path.join(project_root, 'data', 'combined_data.csv')
    
    df_processed = load_and_preprocess_data(data_path)
    run_committee_training(df_processed)
