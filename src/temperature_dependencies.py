import numpy as np
import matplotlib.pyplot as plt

def water_density(T):
    """Water Density. Calculates the density of water at a given temperature. [kg/m^3] """
    return 1000.6 - 0.0128 * T**1.76

def water_specific_heat(T):
    """Water Specific Heat. Calculates the specific heat of water at a given temperature. [J/kg*K] """
    return 4209.1 - 132.8 * 1e-2 * T + 143.2 * 1e-4 * T**2




# Temperature range from 0 to 100 degrees Celsius
temperatures = np.linspace(0, 100, 500)

# Calculate water density and specific heat for each temperature
densities = [water_density(T) for T in temperatures]
specific_heats = [water_specific_heat(T) for T in temperatures]

# Plot water density
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(temperatures, densities, label='Water Density')
plt.xlabel('Temperature (°C)')
plt.ylabel('Density (kg/m$^3$)')
plt.title('Water Density vs Temperature')
plt.legend()
plt.grid()

# Plot water specific heat
plt.subplot(1, 2, 2)
plt.plot(temperatures, specific_heats, label='Water Specific Heat', color='orange')
plt.xlabel('Temperature (°C)')
plt.ylabel('Specific Heat (J/kg·K)')
plt.title('Water Specific Heat vs Temperature')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()