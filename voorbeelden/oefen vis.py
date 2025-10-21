import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os 

os.chdir('C:\\Users\\Win11 Pro\\Documents\\Project-2.1\\metingen')

line_value = 300

df = pd.read_csv('fietsventieldopje.csv')
df['X'] = pd.to_numeric(df['X'], errors='coerce')
df['CH1'] = pd.to_numeric(df['CH1'], errors='coerce')
X = np.array(df['X'])
C = np.array(df['CH1'])
S = np.delete(X,0)
V = np.delete(C,0)

print(df)

plt.figure(figsize=(10,6))
plt.plot(S, V, lw=2)
plt.xlabel('X waarde', fontsize=14)
plt.ylabel('CH1 waarde', fontsize=14)
plt.title('Fietsventieldopje Meting', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.tick_params(axis='both', which='major', labelsize=12, width=2)
plt.xlim(200,400)
plt.tight_layout()
plt.show()