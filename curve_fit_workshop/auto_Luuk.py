import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os

os.chdir("C:\\Users\\Win11 Pro\\Documents\\Project-2.1\\curve_fit_ws")

def PosAuto(t,x0,v):
    return x0 + v*t

df = pd.read_csv('Auto.csv')
t = np.array(df['tijd (s)'])
x = np.array(df['positie (m)'])

popt, pcov = opt.curve_fit(PosAuto,t,x)
x0_fit, v_fit = popt

auto = PosAuto(t,x0_fit,v_fit)

plt.figure(0)
plt.scatter(t,x,color='blue')
plt.plot(t,auto,color='red')
plt.xlabel('tijd (s)')
plt.ylabel('positie (m)')
plt.title('positie over tijd')
plt.grid()

plt.show()