import numpy as np
import matplotlib.pyplot as plt

#constants
n_water_532 = 1.332
k = 0.0015
solubillity_glucose = 909 # g/L
solubillity_glucose_gml = solubillity_glucose / 1000 # g/ml
solubillity_glucose_g100ml = solubillity_glucose / 10 # g/100ml

# function
def refractive_index_glucose(concentration, wavelength=532):    
    return n_water_532 + k * concentration

# with function
concentrations_glucose = np.linspace(0, solubillity_glucose, 100)
refractive_indices = refractive_index_glucose(concentrations_glucose)

# with function 2
concentrations_glucose_g100ml = np.linspace(0, solubillity_glucose_g100ml, 100)
refractive_indices_g100ml = refractive_index_glucose(concentrations_glucose_g100ml)

# wikipedia data for comparison
percentage_glucose = np.array([10,20,60]) # %
n_wikipedia_glucose = np.array([1.3477,1.3635,1.4394]) # at 20 C and 589 nm

# for "sugar"
concentrations_sugar = np.array([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85]) # g/100ml
n_wikipedia_sugar = np.array([1.3330,1.3403,1.3479,1.3557,1.3639,1.3723,1.3811,1.3902,1.3997,1.4096,1.4200,1.4307,1.4418,1.4532,1.4651,1.4774,1.4901,1.5033]) # at canesugar
c_lin = np.array([0,40])
n_sug_lin = np.array([1.333,1.3997])

# plotting
#plt.figure(0)
#plt.plot(concentrations_glucose, refractive_indices)
#plt.xlabel('Concentration of Glucose (g/L)')
#plt.ylabel('Refractive Index at 532 nm')
#plt.title('Refractive Index of Glucose Solutions at 532 nm')
#plt.grid()

plt.figure(1)
plt.plot(percentage_glucose,n_wikipedia_glucose,label='Wikipedia Data at 589 nm')
plt.xlabel('Concentration of Glucose (%)')
plt.ylabel('Refractive Index at 589 nm')
plt.title('Refractive Index of Glucose Solutions at 589 nm')
plt.grid()

plt.figure(2)
plt.plot(concentrations_sugar,n_wikipedia_sugar,label='Wikipedia Data for Cane Sugar')
plt.plot(c_lin,n_sug_lin,label='Linear Approximation',linestyle='--')
plt.xlabel('Concentration of Cane Sugar (g/100ml)')
plt.ylabel('Refractive Index')
plt.title('Refractive Index of Cane Sugar Solutions')
plt.grid()

plt.figure(4)
plt.plot(concentrations_glucose_g100ml, refractive_indices_g100ml)
plt.xlabel('Concentration of Glucose (g/100ml)')
plt.ylabel('Refractive Index at 532 nm')
plt.title('Refractive Index of Glucose Solutions at 532 nm (g/100ml)')
plt.grid()

plt.figure(5)
plt.plot(percentage_glucose,n_wikipedia_glucose,label='Wikipedia Data at 589 nm')
plt.plot(concentrations_glucose_g100ml, refractive_indices_g100ml,label='Calculated at 532 nm')
plt.xlabel('Concentration of Glucose (g/100ml)')
plt.ylabel('Refractive Index')
plt.title('Refractive Index of Glucose Solutions at 532 and 589 nm (g/100ml)')
plt.grid()
plt.legend()

plt.show()
