import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

line_value = 3

def golf(t, A, f):
    return A * np.sin(f * 2 * np.pi * t)

t = np.linspace(0, 10, 1000)
y = golf(t, 5, 2)

# Zoek pieken
peaks, _ = find_peaks(y)

# Filter pieken die boven de lijn zitten
peaks_above = peaks[y[peaks] > line_value]

print("Aantal pieken boven de lijn:", len(peaks_above))

# Plot
line = np.full_like(t, line_value)
plt.plot(t, y)
plt.plot(t, line, label="drempel")
plt.plot(t[peaks_above], y[peaks_above], "ro", label="Pieken boven drempel")
plt.legend()
plt.grid()
plt.show()
