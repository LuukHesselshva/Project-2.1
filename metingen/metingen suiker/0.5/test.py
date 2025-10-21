import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
lambda_laser = 532e-9   # m
d_cuvett = 0.25e-3        # m
d_sugar = 1.0e-3         # m
n_cuvett = 1.33
n_air = 1.0

# Model function
def N_fit_suiker(theta_deg, n_sugar):
    theta_rad = np.deg2rad(theta_deg)
    term_cuv = np.sqrt(n_cuvett**2 - n_air**2 * np.sin(theta_rad)**2) - n_air * np.cos(theta_rad)
    term_sug = np.sqrt(n_sugar**2 - n_air**2 * np.sin(theta_rad)**2) - n_air * np.cos(theta_rad)
    N = (2 / lambda_laser) * (
        2*d_cuvett * term_cuv +
        d_sugar * term_sug -
        (2*n_cuvett*d_cuvett + n_sugar*d_sugar - 2*d_cuvett*n_air - d_sugar*n_air)
    )
    return N

# Data
theta_deg = np.array([0, 2, 4, 6, 8])
fringes = np.array([0, 14, 19, 40, 54])

# Fit
popt, pcov = curve_fit(N_fit_suiker, theta_deg, fringes, p0=[1.33])
n_fit, n_fit_err = popt[0], np.sqrt(np.diag(pcov))[0]
print(f"n_sugar = {n_fit:.4f} ± {n_fit_err:.4f}")

# Generate smooth fit line
theta_fit = np.linspace(0, 12, 200)
fit_curve = N_fit_suiker(theta_fit, n_fit)

# Plot
plt.figure(figsize=(8,5))
plt.scatter(theta_deg, fringes, label="Metingen", color="red")
plt.plot(theta_fit, fit_curve, label=f"Fit: n_sugar = {n_fit:.3f} ± {n_fit_err:.3f}", color="blue")
plt.plot(theta_fit, N_fit_suiker(theta_fit, 1.35), "--", label="Vergelijking n=1.35", color="orange")
plt.xlabel("Hoek (°)")
plt.ylabel("Aantal interferentiefranjes N")
plt.title("Fit van brekingsindex suikeroplossing")
plt.legend()
plt.grid(True)
plt.xlim(0, 12)
plt.ylim(0, 100)
plt.show()
