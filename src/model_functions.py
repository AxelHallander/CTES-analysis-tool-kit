import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.ndimage import uniform_filter
import warnings
from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import savgol_filter

# Suppress the RankWarning
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

def replace_odd_datapoints(data, threshold=16,step=1):
    """Replace odd datapoints in a timeseries with the previous value if the previous value is not odd."""
    # Copy the data
    data_new = np.copy(data)
    counter = 0

    # Loop through the data
    for i in range(1, len(data)):
        # Find the index of the odd datapoints
        index =  np.where(abs(data_new[i][:] - data_new[i-1][:]) > threshold)
        if index[0].size > 0: 
            # Replace the odd datapoints with the previous value
            data_new[i][index] = data_new[i-step][index]
            counter += 1

    return data_new, counter

def replace_odd_datapoints_mean(data, threshold=16,step=3):
    """Replace odd datapoints in a timeseries with the previous value if the previous value is not odd."""
    # Copy the data
    data_new = np.copy(data)
    counter = 0

    # Loop through the data
    for i in range(1, len(data)):
        # Find the index of the odd datapoints
        index = np.where(abs(data_new[i][:] - np.mean([data_new[max(0, i-step)][:], data_new[min(len(data)-1, i+step)][:]], axis=0)) > threshold)
        if index[0].size > 0:
            # Replace the odd datapoints with the mean of surrounding points
            for idx in index[0]:
                if i > step and i + step < len(data):
                    data_new[i][idx] = np.mean([data_new[i-step][idx], data_new[i+step][idx]])
                elif i > step:  # If only previous points are available
                    data_new[i][idx] = data_new[i-step][idx]
                elif i + step < len(data):  # If only next points are available
                    data_new[i][idx] = data_new[i+step][idx]
            counter += 1

    return data_new, counter

def adjust_heat_exchanger_storage_side_power(df_power_district, df_power_storage, calc_charge, ratio, printy=True):
    """Adjust the heat exchanger storage side power. During short charge periods the power is not correct and can be adjusted by mimicing the district side power * ratio. """
    
    # Create a copy of the original DataFrame to avoid modifying it directly
    df_power_storage_new = df_power_storage.copy()

    # Find the charge and discharge periods
    c, d, _ = find_charge_discharge(df_power_district)

    # loop through the charge periods to find odd data 
    for i in range(len(c)):
        diff = np.sum(calc_charge[c[i][0]:c[i][1]] - df_power_storage.iloc[c[i][0]:c[i][1]].values)/(c[i][1]-c[i][0])
        
        # If the difference is positive, adjust the storage power
        if diff > 0:
            if printy == True:
                print(f'Charge diff: {round(diff,2)}  at {c[i]}')

            # Ensure the value being assigned has the same shape as the slice
            storage_slice_len = c[i][1] - c[i][0]
            district_slice = df_power_district.iloc[c[i][0]:c[i][1]].values
            if len(district_slice) == storage_slice_len:
                df_power_storage_new.iloc[c[i][0]:c[i][1]] = district_slice * ratio
            else:
                # If lengths mismatch, print a warning and skip this interval
                print(f"Warning: Length mismatch at interval {c[i]} (district: {len(district_slice)}, storage: {storage_slice_len})")
                continue

    # loop through the discharge periods to find odd data
    for i in range(len(d)):
        diff = np.sum(df_power_district.iloc[d[i][0]:d[i][1]].values - df_power_storage_new.iloc[d[i][0]:d[i][1]].values)/(d[i][1]-d[i][0])
        
        # If the difference is negative, adjust the storage power
        if diff < 0:
            if printy == True:
                print(f'Discharge diff: {round(diff,2)}  at {d[i]}')

            # Ensure the value being assigned has the same shape as the slice
            storage_slice_len = d[i][1] - d[i][0]
            district_slice = df_power_district.iloc[d[i][0]:d[i][1]].values
            if len(district_slice) == storage_slice_len:
                df_power_storage_new.iloc[d[i][0]:d[i][1]] = district_slice * ratio
            else:
                # If lengths mismatch, print a warning and skip this interval
                print(f"Warning: Length mismatch at interval {d[i]} (district: {len(district_slice)}, storage: {storage_slice_len})")
                continue

    return df_power_storage_new


def power_heat_exchanger(T_cold, T_hot, Q):
    """Power Heat Exchanger. Calculates the power of a heat exchanger based on the temperature difference and flow rate. [MW] """
    p = (T_hot - T_cold)*Q*water_density((T_hot + T_cold)/2)*water_specific_heat((T_hot + T_cold)/2)/3600/1e6 # [MW]
    return p

def water_density(T):
    """Water Density. Calculates the density of water at a given temperature. [kg/m^3] """
    return 1000.6 - 0.0128 * T**1.76

def water_specific_heat(T):
    """Water Specific Heat. Calculates the specific heat of water at a given temperature. [J/kg*K] """
    return 4209.1 - 132.8 * 1e-2 * T + 143.2 * 1e-4 * T**2

def thermal_energy_new(T_bot_min, T_top_min, T, V, top_layer, top_volume, degree=1):
    """Heat function. Calculates the energy at every instant using the temperature & volume gradient. This includes a varying top layer."""

    # Allocate
    E = np.zeros(len(T))
    T_copy = np.copy(T)

    T_top =np.zeros((len(T_copy),1))

    for t, temp in enumerate(T_copy):
        T_top[t] = temp[-1]

    # remove the last layer from the temperature array
    T_new = T_copy[:, :-1] 
    T_top = T_top.flatten()

    # define top layer
    top_layer_h = top_layer - 23

    # Ensure top_layer_h is non-negative
    top_layer_h = top_layer_h.clip(lower=0)

    # Calculate the index based on the height and step size
    i = (top_layer_h / 0.05).astype(int)

    # Ensure the index does not exceed the length of top_volume1
    i = i[0:np.size(E)]

    # Calculate the volume for the top layer
    V_last = np.array([np.sum(top_volume[:index]) for index in i])

    # Calculate the heat energy at every data point, time and depth level
    if isinstance(T_new, np.ndarray) and not isinstance(T_new[0], float):
        q = (T_new - min_temperature_distribution(T_new[0], T_bot_min, T_top_min, degree)) * V[0:-1] * water_density(T_new) * water_specific_heat(T_new)
    else:
        q = (T_new - min_temperature_distribution(T_new, T_bot_min, T_top_min, degree)) * V[0:-1] * water_density(T_new) * water_specific_heat(T_new)

    q_last = (T_top - T_bot_min) * V_last * water_density(T_top) * water_specific_heat(T_top)

    # Sum up the energy for each depth level
    for i in range(len(q)): 
        E[i] =+ np.sum(q[i]) + q_last[i]

    return E/(3600*1e9) # Convert to GWh


def thermal_energy(T_bot_min, T_top_min, T, V, degree=1):
    """Heat function. Calculates the energy at every instant using the temperature & volume gradient. """

    # Allocate memory for the energy
    E = np.zeros(len(T))

    # Calculate the heat energy at every data point, time and depth level
    if isinstance(T, np.ndarray) and not isinstance(T[0], float):
        q = (T - min_temperature_distribution(T[0], T_bot_min, T_top_min, degree)) * V * water_density(T) * water_specific_heat(T)
    else:
        q = (T - min_temperature_distribution(T, T_bot_min, T_top_min, degree)) * V * water_density(T) * water_specific_heat(T)
        
    # Sum up the energy for each depth level
    for i in range(len(q)): 
        E[i] =+ np.sum(q[i])

    return E/(3600*1e9) # Convert to GWh

def min_temperature_distribution(T, T_bot_min, T_top_min, degree=1):
    """Minimum Temperature Distribution. Calculates the minimum temperature distribution of the water. """
    
    # Constant distribution
    if degree == 0:
        return np.full(len(T), T_bot_min)
    # Linear distribution
    elif degree == 1:
        return np.linspace(T_bot_min, T_top_min, len(T))
    # Polynomial fit - distribution
    elif degree >= 2:
        x = np.linspace(0, 1, len(T))
        coeffs = np.polyfit([0, 1], [T_bot_min, T_top_min], degree)
        return np.polyval(coeffs, np.linspace(0, 1, len(T)))
    # Logarithmic fit - distribution
    elif degree == -1:
        x = np.linspace(1, len(T), len(T))
        log_fit = np.polyfit(np.log(x), [T_bot_min + (T_top_min - T_bot_min) * (i / (len(T) - 1)) for i in range(len(T))], 1)
        return np.polyval(log_fit, np.log(x))
    else:
        raise ValueError("Degree must be >= 0 or -1")
    
def find_temp_dist(T,min_max):
    """Find the actual max/min temperature distribution. """
    row_means = np.mean(T, axis=1)

    if min_max == 'min':
        i = np.argmin(row_means)
        return T[i]
    elif min_max == 'max':
        i = np.argmax(row_means)
        return T[i]

def find_extreme_index(data, mode='max'):
    """Find the time index where the max/min temperature occurs. """
    
    # Flatten the data to find the index of the max/min value
    data = np.array(data)
    
    if mode == 'max':
        flat_index = np.argmax(data)
    elif mode == 'min':
        flat_index = np.argmin(data)
    else:
        raise ValueError("mode must be 'max' or 'min'")
    
    # Convert flat index to row index
    row_index = np.unravel_index(flat_index, data.shape)[0]
    return row_index

def extract_layers(temp_data, layers):
    """Extract specific layers from temperature data."""

    # preallocate the matrix
    layer_matrix = np.zeros((len(temp_data), len(layers)))
    
    # Extract the layers
    for i in range(len(temp_data)):
        for j, layer in enumerate(layers):
            layer_matrix[i, j] = temp_data[i][layer]
    
    return layer_matrix


def func(x, a, b, c, d):
    """Logistic function. """
    return a / (1.0 + np.exp(-c * (x - d))) + b

# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore")  # do not print warnings by genetic algorithm
    try:
        val = func(xData, *parameterTuple)
        return np.sum((yData - val) ** 2.0)
    except Exception as e:
        return np.inf  # Return a large value if the function fails

# generate initial parameter values
def generate_Initial_Parameters():
    """Initial parameter generation. """
    parameterBounds = []
    parameterBounds.append([0.0, 100.0]) # search bounds for a
    parameterBounds.append([-10.0, 0.0]) # search bounds for b
    parameterBounds.append([0.0, 10.0]) # search bounds for c
    parameterBounds.append([0.0, 10.0]) # search bounds for d

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

def logistical_model(xData,yData):
    """Logistical model. """
    # by default, differential_evolution completes by calling curve_fit() using parameter bounds
    geneticParameters = generate_Initial_Parameters()

    # now call curve_fit without passing bounds from the genetic algorithm,
    # just in case the best fit parameters are aoutside those bounds
    fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters)
    print('Fitted parameters:', fittedParameters)

    modelPredictions = func(xData, *fittedParameters) 

    absError = modelPredictions - yData

    SE = np.square(absError) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(yData))
    
    # create data for the fitted equation plot
    xModel = np.linspace(min(xData), max(xData))
    yModel = func(xModel, *fittedParameters)

    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)

    return yModel, xModel


def func(x, a, b, c):
    return a * np.exp(b * x) + c

# function for genetic algorithm to minimize (sum of squared error)
def sumOfSquaredError(parameterTuple):
    warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
    val = func(xData, *parameterTuple)
    return np.sum((yData - val) ** 2.0)

# generate initial parameter values
def generate_Initial_Parameters():
    """Initial parameter generation. """
    parameterBounds = []
    parameterBounds.append([0.0, 100.0]) # search bounds for a
    parameterBounds.append([-10.0, 0.0]) # search bounds for b
    parameterBounds.append([0.0, 10.0]) # search bounds for c

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

def exponential_model(xData,yData):
    """Logistical model. """
    # by default, differential_evolution completes by calling curve_fit() using parameter bounds
    geneticParameters = generate_Initial_Parameters()

    # now call curve_fit without passing bounds from the genetic algorithm,
    # just in case the best fit parameters are aoutside those bounds
    fittedParameters, pcov = curve_fit(func, xData, yData, geneticParameters)
    print('Fitted parameters:', fittedParameters)

    modelPredictions = func(xData, *fittedParameters) 

    absError = modelPredictions - yData

    SE = np.square(absError) # squared errors
    MSE = np.mean(SE) # mean squared errors
    RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(yData))
    
    # create data for the fitted equation plot
    hourly_data_points = 100*365*24
    xModel = np.linspace(min(xData), max(xData),hourly_data_points)
    yModel = func(xModel, *fittedParameters)

    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)

    return yModel, xModel, fittedParameters


# Calculate the power of the ship
def calculate_power(energy, time, measured_charge, measured_discharge, threshold=10):           # ONE CAN CHANGE THIS, FOR DISCHARGE!
    """Calculate the power of a ship thrugh by using the gradient. A threshold is used. Does not work in the intial warming up. """
    # calculate the power through numeric gradient
    power = np.gradient(energy, time)
    
    # allocate memory for the charge, discharge and losses
    calc_charge = np.zeros_like(power)
    calc_discharge = np.zeros_like(power)
    losses = np.zeros_like(power)

    # calculate the charge, discharge and losses
    for i in range(1, len(power)):
        if measured_charge.iloc[i] > 0:
            # charge, for current and previous timepoint
            calc_charge[i] = power[i]
            if power[i-1] > threshold/2:
                calc_charge[i-1] = power[i-1]
        elif  measured_discharge.iloc[i] < 0: # power[i] < -threshold or
            # discharge, for current and previous timepoint
            calc_discharge[i] = power[i]
            if power[i-1] < -threshold/2:
                calc_discharge[i-1] = power[i-1]
        else: #power[i] < 0 and power[i] > -threshold/2:
            losses[i] = power[i]
  
    steady_losses = np.copy(losses)
    # calculate the losses from charge and discharge
    losses += (calc_charge - measured_charge)
    losses += (- calc_discharge + measured_discharge) # HÄR MAN SKA ÄNDRA TECKEN OM MAN VILL GÖRA TVÄRTOM OCH JUSTERA FÖR UTRÄKNADE URLADDNINGEN SKA VARA STÖRRE ÄN MÄTTA
   
   
    return power, calc_charge, calc_discharge, losses, steady_losses # add to check of measuread and calculated power is at the same time instant?

# calculate power from energy directly
def calulate_charge_based_on_energy(meas_power, energy):
    """Calculate charge based on energy. """
    
    # find the indices for charging and discharing
    measured_charge_index = np.where(meas_power > 0)[0]
    measured_discharge_index = np.where(meas_power < 0)[0]

    # alllocate 
    charging_periods = []
    discharging_periods = []
    current_period_charge = [measured_charge_index[0].item()]
    current_period_discharge = [measured_discharge_index[0].item()]

    # find the charging and discharging periods
    for i in range(1, len(measured_charge_index)):
        if measured_charge_index[i] == measured_charge_index[i-1] + 1:
            current_period_charge.append(measured_charge_index[i].item())
        else:
            current_period_charge.append(measured_charge_index[i-1].item())
            
            charging_periods.append(current_period_charge)
            current_period_charge = [measured_charge_index[i].item()]

    for i in range(1, len(measured_discharge_index)):
        if measured_discharge_index[i] == measured_discharge_index[i-1] + 1:
            current_period_discharge.append(measured_discharge_index[i].item())
        else:
            discharging_periods.append(current_period_discharge)
            current_period_discharge = [measured_discharge_index[i].item()]

    # add the last period
    charging_periods.append(current_period_charge)
    discharging_periods.append(current_period_discharge)

    # allocate for the charge and discharge time periods
    charge_time = np.zeros((len(charging_periods), 2))
    discharge_time = np.zeros((len(discharging_periods), 2))

    # find the charge and discharge periods
    for j in range(0, len(charging_periods)-1):
        charge_time[j] = [charging_periods[j][0], charging_periods[j][-1]]

    for j in range(0, len(discharging_periods)-1):
        discharge_time[j] = [discharging_periods[j][0], discharging_periods[j][-1]]

    # allocate for the calculated power
    calc_power = np.zeros((len(energy), 1))

    # add one to the end time for smoothness
    charge_time[:,1] = charge_time[:,1] + 1
    discharge_time[:,1] = discharge_time[:,1] + 1

    # calculate the mean power based on the energy
    for k in range (0, len(charge_time)):
        index = charge_time[k].astype(int)
        calc_power[index[0]:index[1]] = (energy[index[1]] - energy[index[0]])/(index[1]-index[0]) #MWh / h

    for k in range (0, len(discharge_time)):
        index = discharge_time[k].astype(int)
        calc_power[index[0]:index[1]] = (energy[index[1]] - energy[index[0]])/(index[1]-index[0])


    return calc_power

# Smooth and adjust the power
def smooth_and_adjust_power(data,discharge_meas, threshold_factor=1, filter_size=5):
    """
    Smooth outliers in the data using a moving average filter and adjust positive peaks with negative ones.

    Parameters:
    data (array-like): The input data array.
    threshold_factor (float): The factor to determine the threshold for outliers.
    filter_size (int): The size of the moving average filter.

    Returns:
    array-like: The smoothed data with adjusted power values.
    """

    ################ ADJUST FOR POSITIVE LOSSES ################
    data_copy = np.copy(data)
    discharge_meas_copy = np.copy(discharge_meas)

    # Find the positive power values
    pos_power = np.where(data_copy > 0, data_copy, 0)

    # Find the values and indices where pos_power also corresponds to the discharging in discharge_meas
    discharge_indices = np.where(discharge_meas_copy < 0)[0]
    pos_power_indices = np.where(pos_power > 0)[0]

    # Find common indices
    common_indices = np.intersect1d(discharge_indices, pos_power_indices)

    # Create a new vector with the same indices, initialized with zeros
    new_vector = np.zeros_like(pos_power)

    # Save the common values at the common indices in the new vector
    new_vector[common_indices] = pos_power[common_indices]
    # Adjust the positive losses

    data_copy += - 2* new_vector


    ##########################################################
    # Define a threshold to identify outliers
    threshold = threshold_factor * np.std(data_copy)

    # Identify outliers
    mean = np.mean(data_copy)
    outliers = np.abs(data_copy - mean) > threshold
    smoothed_data = np.copy(data_copy)
    length = len(data_copy)

    for i in range(length):
            if smoothed_data[i] > 0:
                # Look for a negative peak within the window before and after the current index
                for j in range(max(0, i - filter_size), min(i + filter_size, length)):
                    if smoothed_data[j] < 0:
                        # Cancel out the peaks
                        adjustment = min(smoothed_data[i], abs(smoothed_data[j]))
                        smoothed_data[i] -= adjustment
                        smoothed_data[j] += adjustment
                        break

    # Apply a smoothing filter (moving average) to the outliers
    smoothed_data[outliers] = uniform_filter(smoothed_data, size=filter_size)[outliers]
    
    
    
    # Smooth the peaks that were not adjusted
    # Find the values and indices where pos_power also corresponds to the discharging in discharge_meas
    discharge_indices = np.where(discharge_meas_copy < 0)[0]
    neg_power_indices = np.where(smoothed_data < discharge_meas_copy)[0]

    # Find common indices
    common_indices = np.intersect1d(discharge_indices, neg_power_indices)
    smoothed_data[common_indices] = uniform_filter(smoothed_data, size=filter_size)[common_indices]


    # Adjust remaining positive peaks with negative ones
    positive_indices = smoothed_data > 0
    negative_indices = smoothed_data < 0

    positive_sum = np.sum(smoothed_data[positive_indices])
    negative_sum = np.sum(smoothed_data[negative_indices])

    if positive_sum > 0 and negative_sum < 0:
        adjustment_factor = positive_sum / abs(negative_sum)
        smoothed_data[negative_indices] *= (1 + adjustment_factor)
        smoothed_data[positive_indices] = 0
    

    # Adjust the smoothed data to preserve the sum
    original_sum = np.sum(data)
    smoothed_sum = np.sum(smoothed_data)
    if smoothed_sum != 0:
        adjustment_factor = original_sum / smoothed_sum
        smoothed_data *= adjustment_factor


    # Adjust the smoothed data to preserve the sum within each window
    # for start in range(0, length, filter_size*2):
    #     end = min(start + filter_size*2, length)
    #     window_original_sum = np.sum(data[start:end])
    #     window_smoothed_sum = np.sum(smoothed_data[start:end])
    #     if window_smoothed_sum != 0:
    #         adjustment_factor = window_original_sum / window_smoothed_sum
    #         smoothed_data[start:end] *= adjustment_factor
        
    #     smoothed_data[start:end][smoothed_data[start:end] > 0] = 0

        
    return smoothed_data

# Calculate the power through the smooth and andjust power
def power_loss_calculation(energy,time,df_power,threshold=1,filter_size=15,smooth=120):
    """Calculate the power loss over time. """
    power_loss_vec = np.gradient(energy, time)


    power_loss_vec_smooth = smooth_and_adjust_power(power_loss_vec,df_power,threshold,filter_size)
    power_loss_vec_smoothest = uniform_filter(power_loss_vec_smooth,smooth)

    if np.sum(power_loss_vec_smooth) - np.sum(power_loss_vec) > 0.0001:
        print(energy.name, np.sum(power_loss_vec_smooth)-np.sum(power_loss_vec))
     

    return power_loss_vec, power_loss_vec_smooth, power_loss_vec_smoothest

# Find the charging,discharging and non periods
def find_charge_discharge(meas_power):
    
    # Find the indices for charging and discharing
    measured_charge_index = np.where(meas_power > 0)[0]
    measured_discharge_index = np.where(meas_power < 0)[0]

    # Find all indices
    all_indices = np.arange(len(meas_power))

    # Find the rest of the indices (neither charge nor discharge)
    rest_indices = np.setdiff1d(all_indices, np.concatenate((measured_charge_index, measured_discharge_index)))

    # alllocate 
    charging_periods = []
    discharging_periods = []
    non_periods = []
    current_period_charge = [measured_charge_index[0].item()]
    current_period_discharge = [measured_discharge_index[0].item()]
    current_period_non = [rest_indices[0].item()]

    # find the charging and discharging periods
    for i in range(1, len(measured_charge_index)):
        if measured_charge_index[i] == measured_charge_index[i-1] + 1:
            current_period_charge.append(measured_charge_index[i].item())
        else:
            #current_period_charge.append(measured_charge_index[i-1].item())
            charging_periods.append(current_period_charge)
            current_period_charge = [measured_charge_index[i].item()]

    for i in range(1, len(measured_discharge_index)):
        if measured_discharge_index[i] == measured_discharge_index[i-1] + 1:
            current_period_discharge.append(measured_discharge_index[i].item())
        else:
            discharging_periods.append(current_period_discharge)
            current_period_discharge = [measured_discharge_index[i].item()]

    for i in range(1, len(rest_indices)):
        if rest_indices[i] == rest_indices[i-1] + 1:
            current_period_non.append(rest_indices[i].item())
        else:
            non_periods.append(current_period_non)
            current_period_non = [rest_indices[i].item()]

    # add the last period
    charging_periods.append(current_period_charge)
    discharging_periods.append(current_period_discharge)
    non_periods.append(current_period_non)

    # allocate for the charge and discharge periods
    charge_time = np.zeros((len(charging_periods), 2))
    discharge_time = np.zeros((len(discharging_periods), 2))
    non_time = np.zeros((len(non_periods), 2))

    # find the charge and discharge periods
    for j in range(0, len(charging_periods)):
        charge_time[j] = [charging_periods[j][0], charging_periods[j][-1]]

    for j in range(0, len(discharging_periods)):
        discharge_time[j] = [discharging_periods[j][0], discharging_periods[j][-1]]

    for j in range(0, len(non_periods)):
        non_time[j] = [non_periods[j][0], non_periods[j][-1]]


    # add one to the end time for smoothness
    charge_time[:,1] = charge_time[:,1] + 2
    discharge_time[:,1] = discharge_time[:,1] + 2

    # add also in the begging
    charge_time[:,0] = charge_time[:,0] - 1
    discharge_time[:,0] = discharge_time[:,0] - 1

    # Convert to integer
    charge_time = charge_time.astype(int)
    discharge_time = discharge_time.astype(int)
    non_time = non_time.astype(int)

    # save the values that will be removed
    cha_mask = np.ones(len(charge_time), dtype=bool)
    dis_mask = np.ones(len(discharge_time), dtype=bool)

    # remove overlapping periods
    for i in range(0, len(charge_time)-1):
        # if the end time of the current period is greater than the start time of the next period
        if charge_time[i][1] >= charge_time[i+1][0]:
            
            # change the end time of the current period to the end time of the next period
            charge_time[i][1] = charge_time[i+1][1]

            # mark the next period for removal
            cha_mask[i+1] = False

    # remove overlapping periods
    for i in range(0, len(discharge_time)-1):
        # if the end time of the current period is greater than the start time of the next period
        if discharge_time[i][1] >= discharge_time[i+1][0]:
            
            # change the end time of the current period to the end time of the next period
            discharge_time[i][1] = discharge_time[i+1][1]

            # mark the next period for removal
            dis_mask[i+1] = False


    charge_time = charge_time[cha_mask]
    discharge_time = discharge_time[dis_mask]
    
    # add one to the end time for smoothness
    charge_time[:,1] = charge_time[:,1] + 1
    discharge_time[:,1] = discharge_time[:,1] + 1
    
    return charge_time, discharge_time, non_time

# Smooth data over the windows
def smooth_and_adjust_power_window(data, filter_size=10):
    """
    Smooth outliers in the data using a moving average filter and adjust positive peaks with negative ones.

    Parameters:
    data (array-like): The input data array.
    threshold_factor (float): The factor to determine the threshold for outliers.
    filter_size (int): The size of the moving average filter.

    Returns:
    array-like: The smoothed data with adjusted power values.
    """

    # Copy data
    data_copy = np.copy(data)
    smoothed_data = np.copy(data_copy)

    # Adjust remaining positive peaks with negative ones
    positive_indices = smoothed_data > 0
    negative_indices = smoothed_data < 0

    # calculate the sums
    positive_sum = np.sum(smoothed_data[positive_indices])
    negative_sum = np.sum(smoothed_data[negative_indices])

    # if there are positive and negative values adjust the negative regarding the positive
    if positive_sum > 0 and negative_sum < 0:
        adjustment_factor = positive_sum / abs(negative_sum)
        smoothed_data[negative_indices] *= (1 + adjustment_factor)
        smoothed_data[positive_indices] = 0
    
    # Smooth the data further
    smoothed_data = uniform_filter(smoothed_data, size=filter_size)

    # Adjust the smoothed data to preserve the sum
    original_sum = np.sum(data)
    smoothed_sum = np.sum(smoothed_data)
    if smoothed_sum != 0:
        adjustment_factor = original_sum / smoothed_sum
        smoothed_data *= adjustment_factor

    return smoothed_data

# Smooth the power losses over windows for above function
def smooth_over_windows(charge, discharge, non_charge, losses, filter_size=5):
    """Smooth the power losses. """

    # Copy the data
    losses_copy = np.copy(losses)
    smoothed_data = np.copy(losses_copy)

    # Smooth the data within the charge, discharge and non periods for each window
    for c in charge:
        smoothed_data[c[0]:c[1]] = smooth_and_adjust_power_window((losses_copy[c[0]:c[1]]), filter_size)

    for d in discharge:
        smoothed_data[d[0]:d[1]] = smooth_and_adjust_power_window((losses_copy[d[0]:d[1]]), filter_size)

    for n in non_charge:
        smoothed_data[n[0]:n[1]] = uniform_filter(smoothed_data[n[0]:n[1]], filter_size)

    return smoothed_data

######################################## FVB FORMULAS #####################################################

def FVB_continuous_loss_formula(T_top, T_bot, T0 = 10, lambdaa = 4):
    """FVB Loss Formula. Continuous loss (power MW). """
    # Constants from FVB
    top_loss = 9876 # W/K (lambda*SurfaceArea)
    bot_loss = 6817 # W/K (lambda*S)
    lambda_reg = 4 # W/m*K
    P = (top_loss*(T_top-T_bot) + bot_loss*(T_bot-T0))*lambdaa/(lambda_reg*1e6) # MW
    return P

def FVB_initial_loss_formula(T_top, T_bot, T0 = 10, lambdaa = 4):
    """FVB Loss Formula. Initial loss (energy GWh). """
    # Constants from FVB
    top_loss = 1237 # MWh/K (rho*c*V)
    bot_loss = 421 # MWh/K (rho*c*V)
    E = (top_loss*(T_top-T_bot) + bot_loss*(T_bot-T0))/1e3 # GW
    return E

############################################################################################################

def calculate_period_means(charge, discharge, non_charge, losses):
    """Smooth the power losses. """

    # Copy the data
    losses_copy = np.copy(losses)
    charge_mean = np.zeros(len(charge))
    discharge_mean = np.zeros(len(discharge))
    non_charge_mean = np.zeros(len(non_charge))
    
    # Smooth the data within the charge, discharge and non periods for each window
    for i, c in enumerate(charge):
        charge_mean[i] = np.mean((losses_copy[c[0]:c[1]])) 

    for i, d in enumerate(discharge):
        discharge_mean[i] = np.mean((losses_copy[d[0]:d[1]]))

    for i, n, in enumerate(non_charge):
        non_charge_mean[i] = np.mean((losses_copy[n[0]:n[1]]))

    return charge_mean, discharge_mean, non_charge_mean 


def calculate_mean_loss_during_discharging(losses, find_discharge, find_steady_losses):
    """Calculate the loss during discharge through the mean values of the loss before and after discharge. """

    # Copy the losses and allocate
    new_loss = np.copy(losses)
    steady_index = []

    # Find the index for the steady losses before a discharge
    for d in find_discharge:
        for j, n in enumerate(find_steady_losses):
            #print(n)
            if n[0] <= d[0] <= n[1]:
                steady_index.append(j)
    
    idx_pre_dis = steady_index

    # Loop over the index of discharges. Calculate the mean loss for the period before and after discharge
    for dis_idx, i in enumerate(idx_pre_dis):


        # if the period before discharge is too short, use the previous steady loss. Calculate the mean loss for the period before discharge
        if i + 1 < len(find_steady_losses) and find_steady_losses[i][1] - find_steady_losses[i][0] > 2:
            mean_loss_pre_discharge = np.mean(new_loss[find_steady_losses[i][0]:find_steady_losses[i][1]])
            if np.isnan(mean_loss_pre_discharge):
   
                # Calculate without NaN values
                new_loss[find_steady_losses[i][0]:find_steady_losses[i][1]] = np.nan_to_num(new_loss[find_steady_losses[i][0]:find_steady_losses[i][1]])
                mean_loss_pre_discharge = np.mean(new_loss[find_steady_losses[i][0]:find_steady_losses[i][1]][new_loss[find_steady_losses[i][0]:find_steady_losses[i][1]] != 0])
 
        elif i + 2 < len(find_steady_losses) and find_steady_losses[i-1][1] - find_steady_losses[i-1][0] > 2:
            mean_loss_pre_discharge = np.mean(new_loss[find_steady_losses[i][0]:find_steady_losses[i][1]])
            if np.isnan(mean_loss_pre_discharge):

                # Calculate without NaN values
                new_loss[find_steady_losses[i-1][0]:find_steady_losses[i-1][1]] = np.nan_to_num(new_loss[find_steady_losses[i-1][0]:find_steady_losses[i-1][1]])
                mean_loss_pre_discharge = np.mean(new_loss[find_steady_losses[i-1][0]:find_steady_losses[i-1][1]][new_loss[find_steady_losses[i-1][0]:find_steady_losses[i-1][1]] != 0])

        else:
            mean_loss_pre_discharge = np.mean(new_loss[find_steady_losses[i][0]:find_steady_losses[i][1]])
            if np.isnan(mean_loss_pre_discharge):
                # calculate only with sorunding values
                mean_loss_pre_discharge =  np.mean((new_loss[find_steady_losses[i][0]-3:find_steady_losses[i][1]+3]))

        # if the period after discharge is too short, use the next steady loss. Calculate the mean loss for the period after discharge
        if i + 1 < len(find_steady_losses) and find_steady_losses[i+1][1] - find_steady_losses[i+1][0] > 2:
            mean_loss_post_discharge = np.mean(new_loss[find_steady_losses[i+1][0]:find_steady_losses[i+1][1]])
            if np.isnan(mean_loss_post_discharge):
                print('NaN value found in mean loss calculation 4')
        elif i + 2 < len(find_steady_losses) and find_steady_losses[i+2][1] - find_steady_losses[i+2][0] > 2:
            mean_loss_post_discharge = np.mean(new_loss[find_steady_losses[i+2][0]:find_steady_losses[i+2][1]])
            if np.isnan(mean_loss_post_discharge):
                print('NaN value found in mean loss calculation 7')
        else:
            mean_loss_post_discharge = np.mean(new_loss[find_steady_losses[i][0]:find_steady_losses[i][1]])
            if np.isnan(mean_loss_post_discharge):
                # calculate only with sorunding values
                mean_loss_post_discharge = np.mean((new_loss[find_steady_losses[i][0]-3:find_steady_losses[i][1]+3]))

        # Set the mean loss for the period during discharge
        new_loss[find_discharge[dis_idx][0]:find_discharge[dis_idx][1]] = (mean_loss_pre_discharge+mean_loss_post_discharge)/2

    return new_loss 

def calculate_mean_loss_during_charging(losses, find_discharge, find_steady_losses):
    """Calculate the loss during charge through the mean values of the loss after charge. """

    # Copy the losses and allocate
    new_loss = np.copy(losses)
    steady_index = []

    # Find the index for the steady losses before a discharge
    for d in find_discharge:
        for j, n in enumerate(find_steady_losses):
            #print(n)
            if n[0] <= d[0] <= n[1]:
                steady_index.append(j)
    
    idx_pre_dis = steady_index

    # Loop over the index of discharges. Calculate the mean loss for the period before and after discharge
    for dis_idx, i in enumerate(idx_pre_dis):


        # if the period after discharge is too short, use the next steady loss. Calculate the mean loss for the period after discharge
        if i + 1 < len(find_steady_losses) and find_steady_losses[i+1][1] - find_steady_losses[i+1][0] > 2:
            mean_loss_post_discharge = np.mean(new_loss[find_steady_losses[i+1][0]:find_steady_losses[i+1][1]])
            if np.isnan(mean_loss_post_discharge):
                print('NaN value found in mean loss calculation 4')
        elif i + 2 < len(find_steady_losses) and find_steady_losses[i+2][1] - find_steady_losses[i+2][0] > 2:
            mean_loss_post_discharge = np.mean(new_loss[find_steady_losses[i+2][0]:find_steady_losses[i+2][1]])
            if np.isnan(mean_loss_post_discharge):
                print('NaN value found in mean loss calculation 7')
        else:
            mean_loss_post_discharge = np.mean(new_loss[find_steady_losses[i][0]:find_steady_losses[i][1]])
            if np.isnan(mean_loss_post_discharge):
                # calculate only with sorunding values
                mean_loss_post_discharge = np.mean((new_loss[find_steady_losses[i][0]-3:find_steady_losses[i][1]+3]))

        # Set the mean loss for the period during discharge
        new_loss[find_discharge[dis_idx][0]:find_discharge[dis_idx][1]] = mean_loss_post_discharge

    return new_loss 
########################## METHODS FOR CALCULATING LOSSES ###########################################

def GRADIENT_FIRST_STEADY_LOSSES(energy, time_vec, charge_meas, discharge_meas, df_power, adj=[1,1]):
    """Calculate through gradient first method. """

    # Power calculations 
    _, _, _, losses, steady_losses = calculate_power(energy, time_vec, adj[0] * charge_meas, adj[1] * discharge_meas)
   
    # Find the indicies of different stages
    find_charge, find_discharge, find_steady_losses = find_charge_discharge(df_power)

    # Calculate the mean loss around discharging
    gradient_first = calculate_mean_loss_during_discharging(losses, find_discharge, find_steady_losses)

    # First smoothing
    gradient_first = smooth_over_windows(find_charge, find_discharge, find_steady_losses, gradient_first, filter_size=25)

    # Final smoothing
    gradient_first = smooth_and_adjust_power(gradient_first, adj[1] * discharge_meas, threshold_factor=1, filter_size=25)

    # Reduce noice in the losses
    gradient_first = savgol_filter(gradient_first, window_length=72, polyorder=3)

    # Calculate the cumulative energy loss
    cum_energy_loss = gradient_first.cumsum()

    return cum_energy_loss, gradient_first
    

def GRADIENT_FIRST_STEADY_LOSSES_ALL(energy, time_vec, charge_meas, discharge_meas, df_power, adj=[1,1]):
    """Calculate through gradient first method. """

    # Power calculations 
    _, _, _, _, steady_losses = calculate_power(energy, time_vec, adj[0] * charge_meas, adj[1] * discharge_meas)
   
    # Find the indicies of different stages
    find_charge, find_discharge, find_steady_losses = find_charge_discharge(df_power)

    # Calculate the mean loss around discharging
    gradient_first = calculate_mean_loss_during_discharging(steady_losses, find_discharge, find_steady_losses)

    # Calculate the mean loss around charging
    gradient_first = calculate_mean_loss_during_charging(gradient_first, find_charge, find_steady_losses)

    # First smoothing
    gradient_first = smooth_over_windows(find_charge, find_discharge, find_steady_losses, gradient_first, filter_size=25)

    # Final smoothing
    gradient_first = smooth_and_adjust_power(gradient_first, adj[1] * discharge_meas, threshold_factor=1, filter_size=25)

    # Reduce noice in the losses
    gradient_first = savgol_filter(gradient_first , window_length=72, polyorder=3)

    # Calculate the cumulative energy loss
    cum_energy_loss = gradient_first.cumsum()

    return cum_energy_loss, gradient_first


def GRADIENT_SECOND_METHOD(energy, cum_charge_meas, cum_discharge_meas, discharge_meas, time_vec, adj=[1,1]):
    """Calculate through gradient second method. Adj is the adjusting constants for charging and dischargin."""

    # Calculate cumulative energy loss
    cum_energy_loss = energy -  adj[0] * cum_charge_meas - adj[1] * cum_discharge_meas - energy[0]

    # Calculate the power of the energy loss
    power_loss_vec, power_loss_vec_smooth, power_loss_vec_smoothest = power_loss_calculation(cum_energy_loss, time_vec, discharge_meas, smooth=24*5)

    return cum_energy_loss, power_loss_vec, power_loss_vec_smooth, power_loss_vec_smoothest


def GRADIENT_FIRST_METHOD(energy, time_vec, charge_meas, discharge_meas, df_power, adj=[1,1]):
    """Calculate through gradient first method. """

    # Power calculations 
    _, _, _, losses, steady_losses = calculate_power(energy, time_vec, adj[0] * charge_meas, adj[1] * discharge_meas)
   
    # Find the indicies of different stages
    find_charge, find_discharge, find_steady_losses = find_charge_discharge(df_power)

    # Reduce noice in the losses
    gradient_first = savgol_filter(losses , window_length=72, polyorder=3)

    # First smoothing
    gradient_first = smooth_over_windows(find_charge, find_discharge, find_steady_losses, gradient_first, filter_size=25)

    # Final smoothing
    gradient_first = smooth_and_adjust_power(gradient_first, adj[1] * discharge_meas, threshold_factor=1, filter_size=25)

    # Calculate the cumulative energy loss
    cum_energy_loss = gradient_first.cumsum()

    return cum_energy_loss, gradient_first, steady_losses


def GRADIENT_FIRST_COMBINED(energy, time_vec, charge_meas, discharge_meas, df_power, start_time, adj=[1,1]):
    """Calculate through gradient first method. """
    
    # Devide the data into two parts
    beg = slice(0, start_time)
    end = slice(start_time, len(energy))

    cum_energy_loss_beg, gradient_beg, _ = GRADIENT_FIRST_METHOD(energy[beg], time_vec[beg], charge_meas[beg], discharge_meas[beg], df_power.iloc[beg], adj)
    cum_energy_loss_end , gradient_end = GRADIENT_FIRST_STEADY_LOSSES_ALL(energy[end], time_vec[end], charge_meas[end], discharge_meas[end], df_power.iloc[end], adj)

    cum_energy_loss = np.concatenate((cum_energy_loss_beg, cum_energy_loss_end + cum_energy_loss_beg[-1]))
    gradient_first = np.concatenate((gradient_beg, gradient_end))

    return cum_energy_loss, gradient_first



def ENERGY_DIFF_METHOD(df_power, energy, steady_losses, cum_charge_meas, cum_discharge_meas, adj=[1,1]):
    """Calculate power based on energy difference. """

    # Use function to calculate the effective power (charge and discharge) 
    effective_calc_power_mean = calulate_charge_based_on_energy(df_power, energy)

    # Find the effective charge & discharge
    charge_calc_eff = np.where(effective_calc_power_mean > 0, effective_calc_power_mean, 0)
    discharge_calc_eff = np.where(effective_calc_power_mean < 0, effective_calc_power_mean, 0)

    # Calculate the cumulative charge & discharge & steady losses
    cum_charge_calc_eff = charge_calc_eff.cumsum()
    cum_discharge_calc_eff = discharge_calc_eff.cumsum()
    cum_steady_losses = steady_losses.cumsum()

    # Calculate the cumulative energy loss
    cum_loss_during_charging = (-cum_charge_meas*adj[0] + cum_charge_calc_eff)
    cum_loss_during_discharging = (cum_discharge_meas*adj[1] - cum_discharge_calc_eff)
    
    # Total cumulative loss
    cum_loss_ENERGY_DIFF = (cum_loss_during_charging + cum_loss_during_discharging + cum_steady_losses)/1000

    cum_loss_ENERGY_DIFF = smooth_cumulative_loss(cum_loss_ENERGY_DIFF, search_back=150)

    #cum_loss_ENERGY_DIFF = savgol_filter(cum_loss_ENERGY_DIFF , window_length=72, polyorder=3)
    
    return cum_loss_ENERGY_DIFF # ALSO CALCULATE THE POWER LOSS VECTOR, or effective_calc_power_mean


def smooth_cumulative_loss(E, search_back=150):
    """ Function to ilimate bumps and smooth data. Finds regions where the derivative is positive (i.e., the curve is incorrectly increasing).
Searches backward from those points to find a point where the derivative is strongly negative (indicating the start of the incorrect bump).
Applies linear interpolation between the negative spike and the end of the positive bump. """

    # start up
    E = E.copy()
    dE = np.gradient(E)
    i = 150

    while i < len(dE) - 1:
        # Step 1: Find the start of a positive bump
        if dE[i] > 0.002:
            start_pos = i
            # Find the end of the positive bump
            end_pos = i
            while end_pos < len(dE) - 1 and dE[end_pos] > 0:
                end_pos += 1 # last point

            # Step 2: Look back from start_pos up to `search_back` steps for the most negative slope
            back_start = max(0, start_pos - search_back)
            window = dE[back_start:start_pos]
            if len(window) == 0:
                i += 1
                continue
            
            # Find the index for highest neg derivitive
            min_index = np.argmin(window)
            start_interp = back_start + min_index

            # Step 3: Linear interpolation
            if start_interp < end_pos:
                x = np.array([start_interp, end_pos])
                y = np.array([E[start_interp], E[end_pos]])
                interp = np.interp(np.arange(start_interp, end_pos + 1), x, y)
                E[start_interp-5:end_pos + 1 -5] = interp

                # Update derivative and index
                dE = np.gradient(E)
                i = end_pos + 1
            else:
                i += 1
        else:
            i += 1
        
    # Apply smoothing filter
    E = pd.Series(uniform_filter(E, 48))
    return E

def calculate_characteristic_time(V):
    """ Calculate the characteristic time for a given volume of water in a tank. """
    c = 4.186e3 # J/kg/K
    rho = 1000 # kg/m^3
    C = rho * V * c # Convert to kg
    h = 4 # W/m^2/K
    A = 23*300*2 + 15*23*2 + 300*15 # estimate
    tau = C /(h * A * 3600 * 24) # convert to days
    return tau

def estimate_characteristic_time(signal, dt):
    """ Estimate the characteristic time of a signal using autocorrelation. """
    # Normalize the signal
    signal = signal - np.mean(signal)
    ac = np.correlate(signal, signal, mode='full')
    ac = ac[ac.size // 2:]  # keep positive lags
    ac /= ac[0]  # normalize

    # Find when autocorrelation falls below 1/e
    idx = np.where(ac < 1/np.e)[0]
    if len(idx) == 0:
        return None  # no characteristic decay
    return idx[0] * dt


# Define the linear model
def linear_model(energy, k):
    """Linear model for power loss calculation. m is the total calculated continuation loss divided by three ships."""
    m = fvb_cont_loss = FVB_continuous_loss_formula(95, 45)/3
    return k * energy #- m

# Fit polynomial regression model
def polynomial_model(x, y, degree):
    p = Polynomial.fit(x, y, degree)
    return p

# Automized function for fitting the power loss
def power_loss_fit(energy, power_loss,power_loss_predicted_normalized):
    """Fit the power loss to the energy. """

    # Calculate the smoothed loss using uniform filter
    smoothed_loss = np.copy(power_loss)

    # Apply a uniform filter to smooth the data
    smoothed_loss = uniform_filter(smoothed_loss, 120)

    # improve the fit by removing the first part of the data
    dates = slice(4000,len(smoothed_loss)) 

    # apply curve fit
    params, covariance = curve_fit(linear_model, energy[dates], smoothed_loss[dates])

    # Calculate the fitted loss using the fitted parameters
    k1 = params[0]*power_loss_predicted_normalized[0:len(energy)]

    # calcualte the linear fit
    m = FVB_continuous_loss_formula(95, 45)/3
    fitted_loss = linear_model(energy, k1[:]) - m

    # Compute Pearson and Spearman correlations for each ship
    pearson_corr, _ = pearsonr(energy[dates], smoothed_loss[dates])
    spearman_corr, _ = spearmanr(energy[dates], smoothed_loss[dates])

    print(f'Pearson correlation:  {round(pearson_corr, 2)}')
    print(f'Spearman correlatio: {round(spearman_corr, 2)}')

    return fitted_loss, smoothed_loss, pearson_corr, spearman_corr


def power_loss_fit_smooth(energy, power_loss,power_loss_predicted_normalized):
    """Fit the power loss to the energy with smoothing through characteristic time."""

    # Calculate the smoothed loss using uniform filter
    smoothed_loss = np.copy(power_loss)

    # Apply a uniform filter to smooth the data
    #smoothed_loss = uniform_filter(smoothed_loss, 120)

    ##################### Add characteristic time estimation smoothing
    tau_est = estimate_characteristic_time(smoothed_loss, 1)

    smoothed_loss = gaussian_filter1d(smoothed_loss, sigma=tau_est)

    tau_est = estimate_characteristic_time(energy, 1)/5
    tau_est = 70

    smoothed_energy = energy #gaussian_filter1d(energy, sigma=tau_est)
    #####################

    # improve the fit by removing the first part of the data
    dates = slice(4000,len(smoothed_loss)) 

    # apply curve fit
    params, covariance = curve_fit(linear_model, smoothed_energy[dates], smoothed_loss[dates])

    # Calculate the fitted loss using the fitted parameters
    k1 = params[0] * (power_loss_predicted_normalized[0:len(energy)])

    # calcualte the linear fit
    m = FVB_continuous_loss_formula(95, 45)/3
    fitted_loss = linear_model(smoothed_energy, k1[:]) - m
    fitted_loss = gaussian_filter1d(fitted_loss, tau_est/3) 

    # Compute Pearson and Spearman correlations for each ship
    pearson_corr, _ = pearsonr(smoothed_energy[dates]*power_loss_predicted_normalized[dates], smoothed_loss[dates])
    spearman_corr, _ = spearmanr(smoothed_energy[dates]*power_loss_predicted_normalized[dates], smoothed_loss[dates])
    print(f'Pearson correlation:  {round(pearson_corr, 2)}')
    print(f'Spearman correlatio:  {round(spearman_corr, 2)}')

    loss_diff = np.sum(smoothed_loss[dates] - fitted_loss[dates])
    print(f'Loss difference:      {np.round(np.mean(loss_diff), 2)} MW') 

    return fitted_loss, smoothed_loss, pearson_corr, spearman_corr, loss_diff

def power_loss_fit_ENERGY_DIFF(cum_loss_energy_diff, energy, temp_time_hours, charge_measured, discharge_measured, df_power, power_loss_predicted_normalized, filter_size=25, threshold_factor=1):
    """
    Perform energy difference method with all steps included.
    """
    when = slice(0, len(cum_loss_energy_diff)-29)
    power_loss, _, _, _, _ = calculate_power(cum_loss_energy_diff[when] * 1000, temp_time_hours[when], charge_measured[when], discharge_measured[when])

    # Find the indices of different stages
    find_charge, find_discharge, find_steady_losses = find_charge_discharge(df_power.iloc[when])

    # First smoothing
    power_loss = smooth_over_windows(find_charge, find_discharge, find_steady_losses, power_loss, filter_size=filter_size)

    # Final smoothing
    power_loss = smooth_and_adjust_power(power_loss, discharge_measured[when], threshold_factor=threshold_factor, filter_size=15)

    # Fit the power loss
    fitted_loss, smoothest, pearson_corr, spearman_corr, loss_diff = power_loss_fit_smooth(energy[when], power_loss, power_loss_predicted_normalized)

    return power_loss, fitted_loss, smoothest, pearson_corr, spearman_corr, loss_diff


def adjust_odd_datapoints(temp_data,power, comparison_charge, threshold=10):
    """Adjust odd datapoints in a timeseries with the previous value if the previous value is not odd."""
    # Copy the data
    data_new = np.copy(temp_data)
    count = 0
    # Loop through the data
    for t in range(1, len(temp_data)):
        
        if comparison_charge[t] == 0 and power[t] > 0:
            data_new[t] = data_new[t-1]
            count += 1

        elif comparison_charge[t] == 0 and power[t] < -threshold:
            data_new[t] = data_new[t-1]
            count += 1
            
    return data_new, count

def level_temperature(T, level):
    """Level Temperature. Calculates the temperature of a specific level. """
    temp_level = np.zeros(len(T))

    # Get the temperature of one leve
    for i in range(len(T)):
        temp_level[i] = T[i][level]
    return temp_level


def stratification_analysis(T):
    """Stratification Analysis. Calculates the stratification of the ship. """
    
    # Calculate the gradient of the temperature
    gradient = np.gradient(T, axis=1)

    # Pre allocate
    max_gradient = np.zeros(len(gradient))
    min_gradient = np.zeros(len(gradient))
    thermocline_size = np.zeros(len(gradient))
    thermocline_start = np.zeros(len(gradient))

    # loop for each time step
    for g in range(len(gradient)):
        # Calculate the maximum and minimum gradient for each time step
        max_gradient[g] = np.max(gradient[g])
        min_gradient[g] = np.min(gradient[g])
    
        # find where the gradient is greater than half the maximum gradient
        thermocline_index = np.where(gradient[g] > max_gradient[g]/2)[0]
        
        # Identify groups of consecutive indices
        consecutive_groups = np.split(thermocline_index, np.where(np.diff(thermocline_index) != 1)[0] + 1)

        # Find the largest group of consecutive indices
        largest_group = max(consecutive_groups, key=len)

        thermocline_size[g] = len(np.array(largest_group))
        thermocline_start[g] = largest_group[0]

    return thermocline_size, thermocline_start


######################################## Energy Give back formulas ######################################################
# Calculate the power of the ship
def calculate_power1(energy, time, measured_charge, measured_discharge, threshold=10):        
    """Calculate the power of a ship thrugh by using the gradient. A threshold is used. Does not work in the intial warming up. """
    # calculate the power through numeric gradient
    power = np.gradient(energy, time)
    
    # allocate memory for the charge, discharge and losses
    calc_charge = np.zeros_like(power)
    calc_discharge = np.zeros_like(power)
    losses = np.zeros_like(power)

    # calculate the charge, discharge and losses
    for i in range(1, len(power)):
        if measured_charge.iloc[i] > 0:
            # charge, for current and previous timepoint
            calc_charge[i] = power[i]
            if power[i-1] > threshold/2:
                calc_charge[i-1] = power[i-1]
        elif  measured_discharge.iloc[i] < 0: # power[i] < -threshold or
            # discharge, for current and previous timepoint
            calc_discharge[i] = power[i]
            if power[i-1] < -threshold/2:
                calc_discharge[i-1] = power[i-1]
        else: #power[i] < 0 and power[i] > -threshold/2:
            losses[i] = power[i]
  
    steady_losses = np.copy(losses)
    # calculate the losses from charge and discharge
    losses += (calc_charge - measured_charge)
    losses += (calc_discharge - measured_discharge) # ändrat till rätt
   
   
    return power, calc_charge, calc_discharge, losses, steady_losses # add to check of measuread and calculated power is at the same time instant?

def smooth_and_adjust_power1(data,discharge_meas, threshold_factor=1, filter_size=5):
    """
    Smooth outliers in the data using a moving average filter and adjust positive peaks with negative ones.

    Parameters:
    data (array-like): The input data array.
    threshold_factor (float): The factor to determine the threshold for outliers.
    filter_size (int): The size of the moving average filter.

    Returns:
    array-like: The smoothed data with adjusted power values.
    """

    ################ ADJUST FOR POSITIVE LOSSES ################
    data_copy = np.copy(data)
    discharge_meas_copy = np.copy(discharge_meas)

    # Find the positive power values
    pos_power = np.where(data_copy > 0, data_copy, 0)

    # Find the values and indices where pos_power also corresponds to the discharging in discharge_meas
    discharge_indices = np.where(discharge_meas_copy < 0)[0]
    pos_power_indices = np.where(pos_power > 0)[0]

    # Find common indices
    common_indices = np.intersect1d(discharge_indices, pos_power_indices)

    # Create a new vector with the same indices, initialized with zeros
    new_vector = np.zeros_like(pos_power)

    # Save the common values at the common indices in the new vector
    new_vector[common_indices] = pos_power[common_indices]
    # Adjust the positive losses

    data_copy += - 2* new_vector


    ##########################################################
    # Define a threshold to identify outliers
    threshold = threshold_factor * np.std(data_copy)

    # Identify outliers
    mean = np.mean(data_copy)
    outliers = np.abs(data_copy - mean) > threshold
    smoothed_data = np.copy(data_copy)
    length = len(data_copy)

    for i in range(length):
            if smoothed_data[i] > 0:
                # Look for a negative peak within the window before and after the current index
                for j in range(max(0, i - filter_size), min(i + filter_size, length)):
                    if smoothed_data[j] < 0:
                        # Cancel out the peaks
                        adjustment = min(smoothed_data[i], abs(smoothed_data[j]))
                        smoothed_data[i] -= adjustment
                        smoothed_data[j] += adjustment
                        break

    # Apply a smoothing filter (moving average) to the outliers
    smoothed_data[outliers] = uniform_filter(smoothed_data, size=filter_size)[outliers]
    
    
    
    # Smooth the peaks that were not adjusted
    # Find the values and indices where pos_power also corresponds to the discharging in discharge_meas
    discharge_indices = np.where(discharge_meas_copy < 0)[0]
    neg_power_indices = np.where(smoothed_data < discharge_meas_copy)[0]

    # Find common indices
    common_indices = np.intersect1d(discharge_indices, neg_power_indices)
    smoothed_data[common_indices] = uniform_filter(smoothed_data, size=filter_size)[common_indices]


    # Adjust remaining positive peaks with negative ones
    positive_indices = smoothed_data > 0
    negative_indices = smoothed_data < 0

    positive_sum = np.sum(smoothed_data[positive_indices])
    negative_sum = np.sum(smoothed_data[negative_indices])

    if positive_sum > 0 and negative_sum < 0:
        adjustment_factor = positive_sum / abs(negative_sum)
        smoothed_data[negative_indices] *= (1 + adjustment_factor)
        smoothed_data[positive_indices] = 0
    

    # Adjust the smoothed data to preserve the sum
    original_sum = np.sum(data)
    smoothed_sum = np.sum(smoothed_data)
    if smoothed_sum != 0:
        adjustment_factor = original_sum / smoothed_sum
        smoothed_data *= adjustment_factor


    # Adjust the smoothed data to preserve the sum within each window
    # for start in range(0, length, filter_size*2):
    #     end = min(start + filter_size*2, length)
    #     window_original_sum = np.sum(data[start:end])
    #     window_smoothed_sum = np.sum(smoothed_data[start:end])
    #     if window_smoothed_sum != 0:
    #         adjustment_factor = window_original_sum / window_smoothed_sum
    #         smoothed_data[start:end] *= adjustment_factor
        
    #     smoothed_data[start:end][smoothed_data[start:end] > 0] = 0

        
    return smoothed_data

def smooth_and_adjust_power_window1(data, filter_size=10):
    """
    Smooth outliers in the data using a moving average filter and adjust positive peaks with negative ones.

    Parameters:
    data (array-like): The input data array.
    threshold_factor (float): The factor to determine the threshold for outliers.
    filter_size (int): The size of the moving average filter.

    Returns:
    array-like: The smoothed data with adjusted power values.
    """

    # Copy data
    data_copy = np.copy(data)
    smoothed_data = np.copy(data_copy)

    # Adjust remaining positive peaks with negative ones
    positive_indices = smoothed_data > 0
    negative_indices = smoothed_data < 0

    # calculate the sums
    positive_sum = np.sum(smoothed_data[positive_indices])
    negative_sum = np.sum(smoothed_data[negative_indices])

    # if there are positive and negative values adjust the negative regarding the positive
    if positive_sum > 0 and negative_sum < 0:
        adjustment_factor = positive_sum / abs(negative_sum)
        smoothed_data[negative_indices] *= (1 + adjustment_factor)
        smoothed_data[positive_indices] = 0
    
    # Smooth the data further
    smoothed_data = uniform_filter(smoothed_data, size=filter_size)

    # Adjust the smoothed data to preserve the sum
    original_sum = np.sum(data)
    smoothed_sum = np.sum(smoothed_data)
    if smoothed_sum != 0:
        adjustment_factor = original_sum / smoothed_sum
        smoothed_data *= adjustment_factor

    return smoothed_data

# Smooth the power losses over windows for above function
def smooth_over_windows1(charge, discharge, non_charge, losses, filter_size=5):
    """Smooth the power losses. """

    # Copy the data
    losses_copy = np.copy(losses)
    smoothed_data = np.copy(losses_copy)

    # Smooth the data within the charge, discharge and non periods for each window
    for c in charge:
        smoothed_data[c[0]:c[1]] = smooth_and_adjust_power_window1((losses_copy[c[0]:c[1]]), filter_size)

    for d in discharge:
        smoothed_data[d[0]:d[1]] = smooth_and_adjust_power_window1((losses_copy[d[0]:d[1]]), filter_size)

    for n in non_charge:
        smoothed_data[n[0]:n[1]] = uniform_filter(smoothed_data[n[0]:n[1]], filter_size)

    return smoothed_data

def GRADIENT_FIRST_METHOD_new(energy, time_vec, charge_meas, discharge_meas, df_power, adj=[1,1]):
    """Calculate through gradient first method. """

    # Power calculations 
    _, _, _, losses, steady_losses = calculate_power1(energy, time_vec, adj[0] * charge_meas, adj[1] * discharge_meas)
   
    # Find the indicies of different stages
    find_charge, find_discharge, find_steady_losses = find_charge_discharge(df_power)

    # Reduce noice in the losses
    gradient_first = savgol_filter(losses, window_length=72, polyorder=3)

    # First smoothing
    gradient_first = smooth_over_windows1(find_charge, find_discharge, find_steady_losses, gradient_first, filter_size=25)

    # Final smoothing
    #gradient_first = smooth_and_adjust_power1(gradient_first, adj[1] * discharge_meas, threshold_factor=1, filter_size=25)
    

    # Calculate the cumulative energy loss
    cum_energy_loss = gradient_first.cumsum()

    return cum_energy_loss, gradient_first

def ENERGY_DIFF_METHOD_new(df_power, energy, steady_losses, cum_charge_meas, cum_discharge_meas, adj=[1,1]):
    """Calculate power based on energy difference. """

    # Use function to calculate the effective power (charge and discharge) 
    effective_calc_power_mean = calulate_charge_based_on_energy(df_power, energy)

    # Find the effective charge & discharge
    charge_calc_eff = np.where(effective_calc_power_mean > 0, effective_calc_power_mean, 0)
    discharge_calc_eff = np.where(effective_calc_power_mean < 0, effective_calc_power_mean, 0)

    # Calculate the cumulative charge & discharge & steady losses
    cum_charge_calc_eff = charge_calc_eff.cumsum()
    cum_discharge_calc_eff = discharge_calc_eff.cumsum()
    cum_steady_losses = steady_losses.cumsum()

    # Calculate the cumulative energy loss
    cum_loss_during_charging = (-cum_charge_meas*adj[0] + cum_charge_calc_eff)
    cum_loss_during_discharging = (-cum_discharge_meas*adj[1] + cum_discharge_calc_eff)
    
    # Total cumulative loss
    cum_loss_ENERGY_DIFF = (cum_loss_during_charging + cum_loss_during_discharging + cum_steady_losses)/1000

    #cum_loss_ENERGY_DIFF = smooth_cumulative_loss(cum_loss_ENERGY_DIFF, search_back=150)

    cum_loss_ENERGY_DIFF = savgol_filter(cum_loss_ENERGY_DIFF , window_length=72, polyorder=3)
    
    return cum_loss_ENERGY_DIFF # ALSO CALCULATE THE POWER LOSS VECTOR, or effective_calc_power_mean

def GRADIENT_FIRST_COMBINED_new(energy, time_vec, charge_meas, discharge_meas, df_power, start_time, adj=[1,1]):
    """Calculate through gradient first method. """
    
    # Devide the data into two parts
    beg = slice(0, start_time)
    end = slice(start_time, len(energy))

    cum_energy_loss_beg, gradient_beg = GRADIENT_FIRST_METHOD_new(energy[beg], time_vec[beg], charge_meas[beg], discharge_meas[beg], df_power.iloc[beg], adj)
    cum_energy_loss_end , gradient_end = GRADIENT_FIRST_STEADY_LOSSES_ALL(energy[end], time_vec[end], charge_meas[end], discharge_meas[end], df_power.iloc[end], adj)

    cum_energy_loss = np.concatenate((cum_energy_loss_beg, cum_energy_loss_end + cum_energy_loss_beg[-1]))
    gradient_first = np.concatenate((gradient_beg, gradient_end))

    return cum_energy_loss, gradient_first

def GRADIENT_SECOND_METHOD_new(energy, cum_charge_meas, cum_discharge_meas, charge_meas, discharge_meas, df_power, time_vec, adj=[1,1]):
    """Calculate through gradient second method. Adj is the adjusting constants for charging and dischargin."""

    # Calculate cumulative energy loss
    cum_energy_loss = energy -  adj[0] * cum_charge_meas - adj[1] * cum_discharge_meas - energy[0]

    # Power calculations 
    power, _, _, _, _ = calculate_power1(cum_energy_loss, time_vec, adj[0] * charge_meas, adj[1] * discharge_meas)
   
    # Find the indicies of different stages
    find_charge, find_discharge, find_steady_losses = find_charge_discharge(df_power)

    # Reduce noice in the losses
    power_loss = savgol_filter(power, window_length=72, polyorder=3)

    # First smoothing
    power_loss = smooth_over_windows1(find_charge, find_discharge, find_steady_losses, power_loss, filter_size=25)

    return cum_energy_loss, power_loss


def power_loss_fit_ENERGY_DIFF_new(cum_loss_energy_diff, energy, temp_time_hours, charge_measured, discharge_measured, df_power, power_loss_predicted_normalized, filter_size=25, threshold_factor=1):
    """
    Perform energy difference method with all steps included.
    """
    when = slice(0, len(cum_loss_energy_diff)-29)
    power_loss, _, _, _, _ = calculate_power(cum_loss_energy_diff[when] * 1000, temp_time_hours[when], charge_measured[when], discharge_measured[when])

    # Find the indices of different stages
    find_charge, find_discharge, find_steady_losses = find_charge_discharge(df_power.iloc[when])

    # First smoothing
    power_loss = smooth_over_windows(find_charge, find_discharge, find_steady_losses, power_loss, filter_size=filter_size)

    # Fit the power loss
    fitted_loss, smoothest, pearson_corr, spearman_corr, loss_diff = power_loss_fit_smooth(energy[when], power_loss, power_loss_predicted_normalized)

    return power_loss, fitted_loss, smoothest, pearson_corr, spearman_corr, loss_diff


####################################### SENSITIVITY ANALYSIS ######################################################

def sensitivity_analysis_NGB(adj_values, energy_skepp, cum_charge_measured, cum_discharge_measured, charge_measured, discharge_measured, temp_time_hours, df_power, steady_losses, title = 'Sensitivity Analysis (no energy give back)'):
    """Perform sensitivity analysis for different adj values."""

    baseline_output_sec, _ , _ , _ = GRADIENT_SECOND_METHOD(energy_skepp, cum_charge_measured, cum_discharge_measured, discharge_measured, temp_time_hours, adj=[1,1])
    baseline_output_sec = baseline_output_sec/1000

    baseline_output_fir, _ , _ = GRADIENT_FIRST_METHOD(energy_skepp, temp_time_hours, charge_measured, discharge_measured, df_power, adj=[1,1])
    baseline_output_fir = baseline_output_fir/1000

    baseline_output_ED = ENERGY_DIFF_METHOD(df_power, energy_skepp, steady_losses, cum_charge_measured, cum_discharge_measured, adj=[1,1])

    baseline_output_SL = GRADIENT_FIRST_STEADY_LOSSES(energy_skepp, temp_time_hours, charge_measured, discharge_measured, df_power,adj=[1,1])
    baseline_output_SL = baseline_output_SL[0]/1000

    results_sec = []
    results_fir = []
    results_ED = []
    results_SL = []

        # Loop through each adj value for GRADIENT_FIRST_METHOD1
    for adj in adj_values:
        output_fir, _ , _ = GRADIENT_FIRST_METHOD(energy_skepp, temp_time_hours, charge_measured, discharge_measured, df_power, adj=[1*adj,1*adj])
        output_fir = output_fir/1000
        delta_fir = output_fir - baseline_output_fir
        results_fir.append((adj, output_fir, delta_fir))

        # Loop through each adj value for ENERGY_DIFF_METHOD1
    for adj in adj_values:
        output_ED = ENERGY_DIFF_METHOD(df_power, energy_skepp, steady_losses, cum_charge_measured, cum_discharge_measured, adj=[1*adj,1*adj])
        delta_ED = output_ED - baseline_output_ED
        results_ED.append((adj, output_ED, delta_ED))

        # Loop through each adj value for GRADIENT_FIRST_STEADY_LOSSES1
    for adj in adj_values:
        output_SL = GRADIENT_FIRST_STEADY_LOSSES(energy_skepp, temp_time_hours, charge_measured, discharge_measured, df_power, adj=[1*adj,1*adj])
        output_SL = output_SL[0]/1000
        delta_SL = output_SL - baseline_output_SL
        results_SL.append((adj, output_SL, delta_SL))
        
        # Loop through each adj value
    for adj in adj_values:
        output_sec= GRADIENT_SECOND_METHOD(energy_skepp, cum_charge_measured, cum_discharge_measured, discharge_measured, temp_time_hours,adj=[1*adj,1*adj])
        output_sec = output_sec[0]/1000
        delta_sec = output_sec - baseline_output_sec
        results_sec.append((adj, output_sec, delta_sec))


    # Extract results for plotting
    adj_vals_fir = [r[0] for r in results_fir]
    outputs_fir = [r[1] for r in results_fir]
    delta_fir = [r[2] for r in results_fir]

    adj_vals_ED = [r[0] for r in results_ED]
    outputs_ED = [r[1] for r in results_ED]
    delta_ED = [r[2] for r in results_ED]

    adj_vals_SL = [r[0] for r in results_SL]
    outputs_SL = [r[1] for r in results_SL]
    delta_SL = [r[2] for r in results_SL]

    # Extract results for plotting
    adj_vals_sec = [r[0] for r in results_sec]
    outputs_sec = [r[1] for r in results_sec]
    delta_sec = [r[2] for r in results_sec]

    ratio_fir = (output_fir[-1] - baseline_output_fir[-1]) / baseline_output_fir[-1] / (adj_values[-1]-1)
    

    ratio_sec = (output_sec.iloc[-1] - baseline_output_sec.iloc[-1]) / baseline_output_sec.iloc[-1]  / (adj_values[-1]-1)
    ratio_ED = (output_ED.iloc[-1] - baseline_output_ED.iloc[-1])  / baseline_output_ED.iloc[-1] / (adj_values[-1]-1)
    ratio_SL = (output_SL[-1] - baseline_output_SL[-1])  / baseline_output_SL[-1] / (adj_values[-1]-1)

    print(f"Uutput percentage change per percentage change in the measured charge and discharge:")
    print(f"Gradient First Method:               {round(ratio_fir,2)} %")
    print(f"Gradient Second Method:              {round(ratio_sec,2)} %")
    print(f"Energy Difference Method:            {round(ratio_ED,2)} %")
    print(f"Gradient First Steady Losses Method: {round(ratio_SL,2)} %")


    # Create a 2x2 subplot for the results
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot the results for GRADIENT_FIRST_METHOD1
    for i in range(len(adj_vals_fir)):
        axs[0, 0].plot(temp_time_hours, outputs_fir[i], label=f'adj = {adj_vals_fir[i]}')
    axs[0, 0].plot(temp_time_hours, baseline_output_fir, label='Baseline', color='black', linewidth=2)
    axs[0, 0].set_ylabel("Energy Loss (GWh)")
    axs[0, 0].set_title(f"Gradient First Method (1:{ratio_fir:.2f} %)")
    axs[0, 0].grid(True)
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].legend()

    # Plot the results for ENERGY_DIFF_METHOD1
    for i in range(len(adj_vals_ED)):
        axs[0, 1].plot(temp_time_hours, outputs_ED[i], label=f'adj = {adj_vals_ED[i]}')
    axs[0, 1].plot(temp_time_hours, baseline_output_ED, label='Baseline', color='black', linewidth=2)
    axs[0, 1].set_title(f"Energy Difference  Method (1:{ratio_ED:.2f} %)")
    axs[0, 1].grid(True)
    axs[0, 1].tick_params(axis='x', rotation=45)
    axs[0, 1].legend()

    # Plot the results for GRADIENT_FIRST_STEADY_LOSSES1
    for i in range(len(adj_vals_SL)):
        axs[1, 0].plot(temp_time_hours, outputs_SL[i], label=f'adj = {adj_vals_SL[i]}')
    axs[1, 0].plot(temp_time_hours, baseline_output_SL, label='Baseline', color='black', linewidth=2)
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Energy Loss (GWh)")
    axs[1, 0].set_title(f"Gradient First Steady Losses Method (1:{ratio_SL:.2f} %)")
    axs[1, 0].grid(True)
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].legend()

    # Plot the results for GRADIENT_SECOND_METHOD1
    for i in range(len(adj_vals_sec)):
        axs[1, 1].plot(temp_time_hours, outputs_sec[i], label=f'adj = {adj_vals_sec[i]}')
    axs[1, 1].plot(temp_time_hours, baseline_output_sec, label='Baseline', color='black', linewidth=2)
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_title(f"Gradient Second Method (1:{ratio_sec:.2f} %)")
    axs[1, 1].grid(True)
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    #plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return output_sec, output_fir, output_ED, output_SL

def sensitivity_analysis_GB(adj_values, energy_skepp, cum_charge_measured, cum_discharge_measured, charge_measured, discharge_measured, temp_time_hours, df_power, steady_losses, start_time=3500, title='Sensitivity Analysis'):
    """Perform sensitivity analysis for different adj values."""

    baseline_output_sec, _ , _ , _ = GRADIENT_SECOND_METHOD(energy_skepp, cum_charge_measured, cum_discharge_measured, discharge_measured, temp_time_hours, adj=[1,1])
    baseline_output_sec = baseline_output_sec/1000

    baseline_output_fir, _ = GRADIENT_FIRST_METHOD_new(energy_skepp, temp_time_hours, charge_measured, discharge_measured, df_power, adj=[1,1])
    baseline_output_fir = baseline_output_fir/1000

    baseline_output_ED = ENERGY_DIFF_METHOD_new(df_power, energy_skepp, steady_losses, cum_charge_measured, cum_discharge_measured, adj=[1,1])

    baseline_output_SL, _ = GRADIENT_FIRST_COMBINED_new(energy_skepp, temp_time_hours, charge_measured, discharge_measured, df_power, start_time, adj=[1,1])
    baseline_output_SL = baseline_output_SL/1000

    results_sec = []
    results_fir = []
    results_ED = []
    results_SL = []

        # Loop through each adj value for GRADIENT_FIRST_METHOD1
    for adj in adj_values:
        output_fir, _  = GRADIENT_FIRST_METHOD_new(energy_skepp, temp_time_hours, charge_measured, discharge_measured, df_power, adj=[1*adj,1*adj])
        output_fir = output_fir/1000
        delta_fir = output_fir - baseline_output_fir
        results_fir.append((adj, output_fir, delta_fir))

        # Loop through each adj value for ENERGY_DIFF_METHOD1
    for adj in adj_values:
        output_ED = ENERGY_DIFF_METHOD_new(df_power, energy_skepp, steady_losses, cum_charge_measured, cum_discharge_measured, adj=[1*adj,1*adj])
        delta_ED = output_ED - baseline_output_ED
        results_ED.append((adj, output_ED, delta_ED))

        # Loop through each adj value for GRADIENT_FIRST_STEADY_LOSSES1
    for adj in adj_values:
        output_SL, _ = GRADIENT_FIRST_COMBINED_new(energy_skepp, temp_time_hours, charge_measured, discharge_measured, df_power, start_time, adj=[1*adj,1*adj])
        output_SL = output_SL/1000
        delta_SL = output_SL - baseline_output_SL
        results_SL.append((adj, output_SL, delta_SL))
        
        # Loop through each adj value
    for adj in adj_values:
        output_sec= GRADIENT_SECOND_METHOD(energy_skepp, cum_charge_measured, cum_discharge_measured, discharge_measured, temp_time_hours,adj=[1*adj,1*adj])
        output_sec = output_sec[0]/1000
        delta_sec = output_sec - baseline_output_sec
        results_sec.append((adj, output_sec, delta_sec))


    # Extract results for plotting
    adj_vals_fir = [r[0] for r in results_fir]
    outputs_fir = [r[1] for r in results_fir]
    delta_fir = [r[2] for r in results_fir]

    adj_vals_ED = [r[0] for r in results_ED]
    outputs_ED = [r[1] for r in results_ED]
    delta_ED = [r[2] for r in results_ED]

    adj_vals_SL = [r[0] for r in results_SL]
    outputs_SL = [r[1] for r in results_SL]
    delta_SL = [r[2] for r in results_SL]

    # Extract results for plotting
    adj_vals_sec = [r[0] for r in results_sec]
    outputs_sec = [r[1] for r in results_sec]
    delta_sec = [r[2] for r in results_sec]

    ratio_fir = (output_fir[-1] - baseline_output_fir[-1]) / baseline_output_fir[-1] / (adj_values[-1]-1)
    

    ratio_sec = (output_sec.iloc[-1] - baseline_output_sec.iloc[-1]) / baseline_output_sec.iloc[-1]  / (adj_values[-1]-1)
    ratio_ED = (output_ED[-1] - baseline_output_ED[-1])  / baseline_output_ED[-1] / (adj_values[-1]-1)
    ratio_SL = (output_SL[-1] - baseline_output_SL[-1])  / baseline_output_SL[-1] / (adj_values[-1]-1)

    print(f"Uutput percentage change per percentage change in the measured charge and discharge:")
    print(f"Gradient First Method:               {round(ratio_fir,2)} %")
    print(f"Gradient Second Method:              {round(ratio_sec,2)} %")
    print(f"Energy Difference Method:            {round(ratio_ED,2)} %")
    print(f"Gradient First Steady Losses Method: {round(ratio_SL,2)} %")


    # Create a 2x2 subplot for the results
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot the results for GRADIENT_FIRST_METHOD1
    for i in range(len(adj_vals_fir)):
        axs[0, 0].plot(temp_time_hours, outputs_fir[i], label=f'adj = {adj_vals_fir[i]}')
    axs[0, 0].plot(temp_time_hours, baseline_output_fir, label='Baseline', color='black', linewidth=2)
    axs[0, 0].set_ylabel("Energy Loss (GWh)")
    axs[0, 0].set_title(f"Gradient First Method (1:{ratio_fir:.2f} %)")
    axs[0, 0].grid(True)
    axs[0, 0].tick_params(axis='x', rotation=45)
    axs[0, 0].legend()

    # Plot the results for ENERGY_DIFF_METHOD1
    for i in range(len(adj_vals_ED)):
        axs[0, 1].plot(temp_time_hours, outputs_ED[i], label=f'adj = {adj_vals_ED[i]}')
    axs[0, 1].plot(temp_time_hours, baseline_output_ED, label='Baseline', color='black', linewidth=2)
    axs[0, 1].set_title(f"Energy Difference  Method (1:{ratio_ED:.2f} %)")
    axs[0, 1].grid(True)
    axs[0, 1].tick_params(axis='x', rotation=45)
    axs[0, 1].legend()

    # Plot the results for GRADIENT_FIRST_STEADY_LOSSES1
    for i in range(len(adj_vals_SL)):
        axs[1, 0].plot(temp_time_hours, outputs_SL[i], label=f'adj = {adj_vals_SL[i]}')
    axs[1, 0].plot(temp_time_hours, baseline_output_SL, label='Baseline', color='black', linewidth=2)
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Energy Loss (GWh)")
    axs[1, 0].set_title(f"Gradient First - Steady Losses Method (1:{ratio_SL:.2f} %)")
    axs[1, 0].grid(True)
    axs[1, 0].tick_params(axis='x', rotation=45)
    axs[1, 0].legend()

    # Plot the results for GRADIENT_SECOND_METHOD1
    for i in range(len(adj_vals_sec)):
        axs[1, 1].plot(temp_time_hours, outputs_sec[i], label=f'adj = {adj_vals_sec[i]}')
    axs[1, 1].plot(temp_time_hours, baseline_output_sec, label='Baseline', color='black', linewidth=2)
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_title(f"Gradient Second Method (1:{ratio_sec:.2f} %)")
    axs[1, 1].grid(True)
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.suptitle('Sensitivity Analysis', fontsize=16, y=1.02)
    #plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.show()

    return output_sec, output_fir, output_ED, output_SL
