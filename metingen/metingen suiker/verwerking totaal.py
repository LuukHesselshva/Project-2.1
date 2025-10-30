import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

n_water_532 = 1.3382
f_sacharose = 1.77e-3
V_totaal = 3

def lin(a,x,b):
    return a*x+b
def concentratie_correctie(V,C):
    return (C/3)*100 

C = np.array([0,0.25,0.5,0.75,1.0])
n = np.array([1.17,1.21,1.26,1.21,1.41])
C_cor = concentratie_correctie(V_totaal,C)
n_theorie = lin(f_sacharose,C_cor,n_water_532)
#print(C_cor)
percentage = np.linspace(0,100,100)
n_percentage = lin(f_sacharose,percentage,n_water_532)

popt,pcov = curve_fit(lin,C,n,)
slope,intercept = popt
trend = lin(slope,C,intercept)

plt.figure(0)
plt.scatter(C_cor,n,label='data',color='blue')
plt.plot(C_cor,trend,label=f'y = {round(slope,4)}x + {round(intercept,4)}',color='red',linestyle='--')
plt.plot(C_cor,n_theorie,label='waarde theoretisch',color='orange')
plt.xlabel('m/v%')
plt.ylabel('n')
plt.legend()
plt.grid()

plt.figure(1)
plt.plot(percentage,n_percentage)
plt.xlabel('m/v%')
plt.ylabel('n')
plt.grid()

plt.show()