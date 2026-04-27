import matplotlib.pyplot as plt
import numpy as np
import os

def _save_plot(model_name, title):
    """Helper function to create directories and save the plot."""
    output_dir = os.path.join('outputs', model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = title.replace(" ", "_").replace(":", "").replace("-", "").replace("#", "").lower()
    save_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_mape_by_day(mape_by_day, model_name, title='MAPE by Day of the Week'):
    """Plots a line chart of MAPE values for each day of the week."""
    week_days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    days = [week_days.get(i, str(i)) for i in sorted(mape_by_day.keys())]
    mape_values = [mape_by_day[i] if mape_by_day[i] is not None else 0 for i in sorted(mape_by_day.keys())]

    plt.figure(figsize=(12, 7))
    plt.plot(days, mape_values, marker='o', linestyle='-', color='skyblue')
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    _save_plot(model_name, title)
    plt.show()

def plot_hourly_mape(hourly_mape, model_name, title='Hourly MAPE'):
    """Plots a line chart of MAPE values for each hour."""
    hours = range(len(hourly_mape))
    mape_values = [float(mape) for mape in hourly_mape]
    
    plt.figure(figsize=(12, 7))
    plt.plot(hours, mape_values, marker='o', linestyle='-', color='lightgreen')
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks(hours)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    _save_plot(model_name, title)
    plt.show()

def plot_hourly_predictions_comparison(y_true, y_pred, model_name, title='Hourly Load Prediction'):
    """Plots a line chart comparing actual and predicted load for each hour."""
    hours = range(len(y_true))
    pred_values = [float(p) for p in y_pred]

    plt.figure(figsize=(12, 7))
    plt.plot(hours, y_true, marker='o', linestyle='-', color='royalblue', label='Actual Load')
    plt.plot(hours, pred_values, marker='x', linestyle='--', color='coral', label='Predicted Load')
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Load', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks(hours)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    _save_plot(model_name, title)
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, model_name, title='Actual vs. Predicted Load'):
    """Plots a line chart comparing actual and predicted values for a 24-hour period."""
    hours = range(len(y_true))
    
    plt.figure(figsize=(12, 7))
    plt.plot(hours, y_true, marker='o', linestyle='-', color='royalblue', label='Actual Load')
    plt.plot(hours, y_pred, marker='x', linestyle='--', color='coral', label='Predicted Load')
    plt.xlabel('Hour', fontsize=12)
    plt.ylabel('Load', fontsize=12)
    plt.title(title, fontsize=16)
    plt.xticks(hours)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    _save_plot(model_name, title)
    plt.show()
