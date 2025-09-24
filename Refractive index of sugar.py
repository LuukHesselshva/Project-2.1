import numpy as np
import matplotlib.pyplot as plt


n_water_532 = 1.332
k = 0.0015
solubillity_glucose = 909 # g/L

def refractive_index_glucose(concentration, wavelength=532):    
    return n_water_532 + k * concentration

concentrations_glucose = np.linspace(0, solubillity_glucose, 100)
refractive_indices = refractive_index_glucose(concentrations_glucose)

plt.plot(concentrations_glucose, refractive_indices)
plt.xlabel('Concentration of Glucose (g/L)')
plt.ylabel('Refractive Index at 532 nm')
plt.title('Refractive Index of Glucose Solutions at 532 nm')
plt.grid()
plt.show()
