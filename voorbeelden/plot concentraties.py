import numpy as np
import matplotlib.pyplot as plt

omega_625nm = 2.81 # L/(mmol*cm)

def lambertbeer(omega,C,d):
    A = omega * C * d
    return A

def lin(a,x,b):
    return a*x + b

C = np.linspace(0,100,1000)
d = 1 # cm
A = lambertbeer(omega_625nm,C,d)

plt.plot(C,A)
plt.xlabel('Concentratie (mmol/L)')
plt.ylabel('Absorbantie (E)')
plt.title('wet van Lambert-Beer')
plt.grid()
plt.show()
