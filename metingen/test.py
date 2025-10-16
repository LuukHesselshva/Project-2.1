import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('C:\\Users\\Win11 Pro\\Documents\\Project-2.1\\metingen')

# Lees CSV met flexibele kolommen
df = pd.read_csv(
    'fietsventieldopje.csv',
    skiprows=2,
    delimiter=',',
    names=['X', 'CH1', 'dummy1', 'dummy2'],
    engine='python'
)

# Gebruik alleen X en CH1
df['X'] = pd.to_numeric(df['X'], errors='coerce')
df['CH1'] = pd.to_numeric(df['CH1'], errors='coerce')
df = df.dropna(subset=['X', 'CH1'])

print(df.head())  # Controleer of er data is

plt.figure(figsize=(12, 6))
plt.plot(df['X'], df['CH1'], lw=2, color='royalblue')
plt.xlabel('Tijd / Sequence', fontsize=14)
plt.ylabel('Spanning (V)', fontsize=14)
plt.title('Oscilloscoopdata Fietsventieldopje', fontsize=16)
plt.grid(True, linestyle='--', linewidth=0.7)
#plt.tight_layout()
#plt.xlim(200,400)
plt.show()