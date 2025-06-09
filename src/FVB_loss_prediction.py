import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Read the operational data from a CSV file
data_power = pd.read_csv('C:/Users/extaxha/Documents/project_data/estimated_power_loss.csv')
starting_value = pd.DataFrame({'year': [0], 'loss': [5]})
data_power = pd.concat([starting_value, data_power], ignore_index=True)

year = data_power['year']
heat_loss = data_power['loss']
contineous_heat_loss = 0.65 # 6.4% of 100GWh from report

# Add a few more data points to the x-axis 1200: each month for 100 years
more_data_points = 365*100*24
fractional_years = np.linspace(min(year), max(year), more_data_points)

# Fit a smooth spline to the filtered data
spline_filtered = UnivariateSpline(year, heat_loss, s=0.03)  # s=0 means interpolation through all points (no smoothing)
predicted_spline_loss = spline_filtered(fractional_years)





