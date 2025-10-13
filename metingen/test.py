import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os 

os.chdir('C:\\Users\\Win11 Pro\\Documents\\Project-2.1\\metingen')

line_value = 300

df = pd.read_csv('test3.csv')
df2 = pd.read_csv('test2.csv')

y = np.array(df['raw'])
x = np.linspace(0, len(y)-1, len(y))

y1 = np.array(df2['raw'])
x1 = np.linspace(0, len(y1)-1, len(y1))

plt.plot(x, y)
plt.plot(x1, y1)
plt.show()