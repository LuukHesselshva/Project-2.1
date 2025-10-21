import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

os.chdir('C:\\Users\\luukj\\Documents\\GitHub\\Project-2.1\\metingen\\metingen suiker\\0.5')

window_size = 5

def read(name):
    df = pd.read_csv(
    str(name) +'.csv',
    skiprows=2,
    delimiter=',',
    names=['X', 'CH1', 'dummy1', 'dummy2'],
    engine='python'
    )
    return df

df02 = read('0.5ml_0g2g')
df24 = read('0.5ml_2g4g')
df46 = read('0.5ml_4g6g')
df68 = read('0.5ml_6g8g')
df810 = read('0.5ml_8g10g')
# Rolling averages toevoegen
df02['rolling'] = df02['CH1'].rolling(window=window_size, center=True).mean()
df24['rolling'] = df24['CH1'].rolling(window=window_size, center=True).mean()
df46['rolling'] = df46['CH1'].rolling(window=window_size, center=True).mean()
df68['rolling'] = df68['CH1'].rolling(window=window_size, center=True).mean()
df810['rolling'] = df810['CH1'].rolling(window=window_size, center=True).mean()

peaks_02, props_02 = find_peaks(df02['rolling'], prominence=0.5)


# Plotten
plt.figure(0)
plt.plot(df02['X'], df02['CH1'], alpha=0.5, label='Origineel')
plt.plot(df02['X'], df02['rolling'], color='red', label=f'Rolling avg ({window_size})')
plt.title('0 → 2')
plt.legend()

plt.figure(1)
plt.plot(df24['X'], df24['CH1'], alpha=0.5, label='Origineel')
plt.plot(df24['X'], df24['rolling'], color='red', label=f'Rolling avg ({window_size})')
plt.title('2 → 4')
plt.legend()

plt.figure(2)
plt.plot(df46['X'], df46['CH1'], alpha=0.5, label='Origineel')
plt.plot(df46['X'], df46['rolling'], color='red', label=f'Rolling avg ({window_size})')
plt.title('4 → 6')
plt.legend()

plt.figure(3)
plt.plot(df68['X'], df68['CH1'], alpha=0.5, label='Origineel')
plt.plot(df68['X'], df68['rolling'], color='red', label=f'Rolling avg ({window_size})')
plt.title('6 → 8')
plt.legend()

plt.figure(4)
plt.plot(df810['X'], df810['CH1'], alpha=0.5, label='Origineel')
plt.plot(df810['X'], df810['rolling'], color='red', label=f'Rolling avg ({window_size})')
plt.title('8 → 10')
plt.legend()

#plt.show()


