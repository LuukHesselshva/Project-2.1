import numpy as np 
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os

# functions
def theta_out(d,n,lambda_i):
    theta = np.arcsin((n*lambda_i)/d)
    return np.rad2deg(theta)
def lambda_out(d,theta,n):
    theta_rad = np.deg2rad(theta)
    return (d*np.sin(theta_rad))/(n)
def theta_patroon(a,L):
    return np.rad2deg(np.arctan(a/L))

# variables vast
d = (1/600)/1000
n = np.array([1,2])
lambda_1 = 490e-9
lambda_2 = 575e-9
# vaste golflengte
theta_1 = theta_out(d,n,lambda_1)
theta_2 = theta_out(d,n,lambda_2)

# data
a = np.array([54.1e-2,129.4e-2])
L = 158.8e-2
theta_data = theta_patroon(a,L)
golf_lengte = lambda_out(d,theta_data,n)
print('golf lengte: ',golf_lengte)
print('gemiddelde:', np.mean(golf_lengte), '±', (golf_lengte[0]-np.mean(golf_lengte)))

# grafiek
plt.figure(0)
plt.plot(n,theta_1,color='red',label='490nm',linestyle='--')
plt.plot(n,theta_2,color='blue',label='575nm',linestyle='--')
plt.scatter(n,theta_data,color='purple',label='data')
plt.plot(n,theta_data,color='purple',linestyle='--')
plt.xlabel('n')
plt.ylabel('theta (graden)')
plt.title('theta bij verschillende n waarden')
plt.grid()
plt.legend()
plt.show()