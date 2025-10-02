# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# constants
lambda_laser = 532e-9 # wavelength of the laser in meters
thickness = 1e-3 # thickness in meters

def N_fit(theta_in,n):
    """Function to fit the refractive index data."""
    theta_in_rad = np.deg2rad(theta_in)
    return ((2*thickness)/lambda_laser) * ( (n**2)/(np.sqrt(n**2-np.sin(theta_in_rad)**2)) - n - np.cos(theta_in_rad) + 1 - (np.sin(theta_in_rad)**(3/2))/n)
def N_fit2(theta_in,n):
    """Alternative function to fit the refractive index data."""
    theta_in_rad = np.deg2rad(theta_in)
    return (2*thickness/lambda_laser) * (np.sqrt(n**2 - np.sin(theta_in_rad)**2) - np.cos(theta_in_rad)+1-n)

i = np.linspace(0,90,1000)
n = 1.5
N = N_fit(i,n)
N2 = N_fit2(i,n)

plt.plot(i,N,label='n=1.5')
plt.plot(i,N2,label='n=1.5 (alt)')
plt.xlabel('Angle of incidence (degrees)')
plt.ylabel('N (fringes)')
plt.title('N vs Angle of Incidence for n=1.5')
plt.xlim(0,10)
plt.ylim(0,50)
plt.legend()
plt.grid()
plt.show()
