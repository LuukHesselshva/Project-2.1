import numpy as np
import matplotlib.pyplot as plt

V_in = 12
V_out = 3


def calculate_R1(R2, V_in, V_out):
    return(V_in*R2/V_out)-R2

R2 = np.linspace(1, 1000, 1000)
R1 = calculate_R1(R2, V_in, V_out)

plt.figure(0)
plt.plot(R2, R1)
plt.xlabel('R2 (Kilo Ohm)')
plt.ylabel('R1 (Kilo Ohm)')
plt.title('R1 als functie van R2 voor een spanningsdeler')
plt.grid()

plt.show()
