import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os

def Spanning(t,V0,tau):
    return V0*(1-np.exp(-t/tau))

os.chdir("C:\\Users\\Win11 Pro\\Documents\\Project-2.1\\curve_fit_ws")
df = pd.read_csv('Condensator.csv')

t = np.array(df['tijd (ms)'])
V = np.array(df['spanning (V)'])

popt,pcov = opt.curve_fit(Spanning,t,V)
tau,V0 = popt
span = Spanning(t,V0,tau)

plt.scatter(t,V,color='blue')
plt.plot(t,span,color='red')
plt.xlabel('tijd (s)')
plt.ylabel('spanning (V)')
plt.title('spanning over tijd')
plt.grid()

plt.show()