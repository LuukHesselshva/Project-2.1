import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os

def Spanning(t,V0,tau):
    return V0*(1-np.exp(-t/tau))
def ms2s(t):
    return t/1000

#os.chdir("C:\\Users\\Win11 Pro\\Documents\\Project-2.1\\curve_fit_ws") #juiste locatie instellen
os.chdir("C:\\Users\\luukj\\Documents\\GitHub\\Project-2.1\\curve_fit_workshop")

df = pd.read_csv('Condensator.csv')

t_ms = np.array(df['tijd (ms)'])
V = np.array(df['spanning (V)'])
t = ms2s(t_ms)

popt,pcov = opt.curve_fit(Spanning,t,V)
V0,tau = popt
span = Spanning(t,V0,tau)

plt.scatter(t,V,color='blue')
plt.plot(t,span,color='red')
plt.xlabel('tijd (s)')
plt.ylabel('spanning (V)')
plt.title('spanning over tijd')
plt.grid()

plt.show()