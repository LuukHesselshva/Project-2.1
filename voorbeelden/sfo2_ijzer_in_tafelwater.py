import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def trendlijn(a,x,b):
    return a*x+b
def C_bereken(a,x,b):
    return (x-b)/a
def stdev(x):
    xmean = np.average(x)
    array_in = x
    array_out = np.zeros_like(array_in)
    n_array = len(array_in)
    for i in range(n_array):
        array_out[i] = (array_in[i]-xmean)**2
    sum_array_out = np.sum(array_out)
    return np.sqrt(sum_array_out/(n_array-1))
    

pipet = np.array([0,5,10,20,30]) # in ml
C_Fe_std = np.array([0.0,0.0712,0.1424,0.2848,0.4272]) # in mg/100ml van std
E_std = np.array([0.0,0.143,0.286,0.569,0.857]) # extinctie std
monster_E = np.array([0.445,0.43,0.44,0.429,0.443,0.429,0.442,0.428,0.44,0.432,0.426,0.425]) # extictie monster

slope, intercept, r_value, p_value, std_err = stats.linregress(C_Fe_std, E_std)
r_squared = r_value **2
trend = trendlijn(C_Fe_std,slope,intercept)

C_monster = C_bereken(slope,monster_E,intercept)
C_monster_gem = np.average(C_monster)
C_monster_stdev = np.std(C_monster)

plot_gem = np.full_like(C_monster,C_monster_gem)
plot_1s = np.full_like(C_monster,(C_monster_gem + C_monster_stdev))
plot_2s = np.full_like(C_monster,(C_monster_gem + 2*C_monster_stdev))
plot_3s = np.full_like(C_monster,(C_monster_gem + 3*C_monster_stdev))
plot_min1s = np.full_like(C_monster,(C_monster_gem - 1*C_monster_stdev))
plot_min2s = np.full_like(C_monster,(C_monster_gem - 2 *C_monster_stdev))
plot_min3s = np.full_like(C_monster,(C_monster_gem - 3*C_monster_stdev))
n = np.linspace(1,len(plot_gem),(len(plot_gem)))

plt.figure(0)
plt.title('ijzer standaard')
plt.xlabel('C Fe (mg/100ml)')
plt.ylabel('E')
plt.grid()
plt.figtext(0.15, 0.83, f'y = {round(slope,4)}x + {round(intercept,4)}',size=8)
plt.figtext(0.15, 0.77, f'R^2 = {round(r_squared,4)}',size=8)
plt.scatter(C_Fe_std,E_std)
plt.plot(C_Fe_std,trend,label='trendlijn',linestyle='--',c='red')

plt.figure(1)
plt.title('shewert')
plt.xlabel('n')
plt.ylabel('concentratie (mg/100ml)')
plt.grid()
plt.figtext(0.15, 0.85, f'gem = {round(C_monster_gem,4)}',size=8)
plt.figtext(0.15, 0.77, f'stdev = {round(C_monster_stdev,4)}',size=8)

plt.scatter(n,C_monster,c='blue',label='data')
plt.plot(n,C_monster,c='blue',label='data',linestyle='--')
plt.plot(n,plot_gem,linestyle='--',c='red',label='average')
plt.plot(n,plot_1s,linestyle='--',c='green',label='1s')
plt.plot(n,plot_2s,linestyle='--',c='yellow',label='2s')
plt.plot(n,plot_3s,linestyle='--',c='purple',label='3s')
plt.plot(n,plot_min1s,linestyle='--',c='green',label='1s')
plt.plot(n,plot_min2s,linestyle='--',c='yellow',label='2s')
plt.plot(n,plot_min3s,linestyle='--',c='purple',label='3s')

plt.show()