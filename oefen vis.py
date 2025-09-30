import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os 

os.chdir('C:\\Users\\Win11 Pro\\Desktop\\oscilloscope rigol\\metingen')

line_value = 300

df = pd.read_csv('Waveform.csv')
S = np.array(df['Sequence'])
V = np.array(df['VOLT'])

# Zoek pieken
peaks, _ = find_peaks(V)

# Filter pieken die boven de lijn zitten
peaks_above = peaks[V[peaks] > line_value]

print("Aantal pieken boven de lijn:", len(peaks_above))

# Plot
line = np.full_like(S, line_value)
plt.plot(S, V)
plt.plot(S, line, label="drempel")
plt.plot(S[peaks_above], V[peaks_above], "ro", label="Pieken boven drempel")
plt.legend()
plt.grid()
plt.show()