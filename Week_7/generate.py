import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
start_date = datetime(2023, 1, 1)
num_days = 365  # One year of data

dates = [start_date + timedelta(days=i) for i in range(num_days)]

# Generate synthetic sales data
sales = np.random.randint(100, 1000, size=num_days)

# Generate weather conditions (temperature in °C, and binary rain indicator)
temperatures = np.random.uniform(5, 35, size=num_days)  # Temperature between 5°C and 35°C
rain = np.random.choice([0, 1], size=num_days, p=[0.7, 0.3])  # 30% chance of rain

# Mark holidays (randomly selecting 10 days as holidays)
holidays = np.zeros(num_days)
holiday_indices = np.random.choice(range(num_days), size=10, replace=False)
holidays[holiday_indices] = 1

# Generate promotions (randomly selecting days with promotions)
promotions = np.random.choice([0, 1], size=num_days, p=[0.8, 0.2])  # 20% chance of promotion

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Sales': sales,
    'Temperature_C': temperatures,
    'Rain': rain,
    'Holiday': holidays,
    'Promotion': promotions
})

# Save to CSV
df.to_csv('sales_data.csv', index=False)

print("Synthetic sales data generated and saved as 'sales_data.csv'.")
