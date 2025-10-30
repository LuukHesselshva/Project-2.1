import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Pad aanpassen aan jouw situatie
main_path = 'C:\\Users\\luukj\\Documents\\GitHub\\Project-2.1\\metingen\\metingen suiker'
os.chdir(main_path)

# constants
lambda_laser = 532e-9  # wavelength of the laser in meters
d_sugar = 1e-2          # thickness in meters
d_cuvett = 1.25e-2 - d_sugar  # thickness in meters
n_cuvett = 1.59         # refractive index of the cuvett
n_air = 1.0             # refractive index of air

n_water_532 = 1.3382
f_sacharose = 1.77e-3
V_totaal = 3

win_size = 3 
prom = 0.24

# Zet op True om piekplots te tonen
check = False

# -------------------------------
# Functies
# -------------------------------

def lin(a, x, b):
    return a * x + b

def concentratie_correctie(V, C):
    return (C / 3) * 100

def read(name, subdir):
    fname = str(name) + '.csv'
    path = os.path.join(subdir, fname) if subdir else fname
    df = pd.read_csv(
        path,
        skiprows=2,
        delimiter=',',
        names=['X', 'CH1', 'dummy1', 'dummy2'],
        engine='python'
    )
    return df

def N_fit_suiker(theta_in, n_sugar):
    """Function to fit the refractive index data."""
    theta_in_rad = np.deg2rad(theta_in)
    return 2 / lambda_laser * (
        2 * d_cuvett * (np.sqrt(n_cuvett**2 - n_air**2 * np.sin(theta_in_rad)**2) - n_air * np.cos(theta_in_rad))
        + d_sugar * (np.sqrt(n_sugar**2 - n_air**2 * np.sin(theta_in_rad)**2) - n_air * np.cos(theta_in_rad))
        - (2 * n_cuvett * d_cuvett + n_sugar * d_sugar - 2 * d_cuvett * n_air - d_sugar * n_air)
    )

def fringes(naam, var_window_size, var_prominence, subdir, show_plots=False):
    """Bepaalt fringes, toont optioneel piekdetectieplots en print samenvatting."""
    naam_1 = str(naam) + '_0g2g'
    naam_2 = str(naam) + '_2g4g'
    naam_3 = str(naam) + '_4g6g'
    naam_4 = str(naam) + '_6g8g'
    naam_5 = str(naam) + '_8g10g'

    dfs = []
    names = [naam_1, naam_2, naam_3, naam_4, naam_5]

    # Lees data in en bereken rolling averages
    for nm in names:
        df = read(nm, subdir)
        df['rolling'] = df['CH1'].rolling(window=var_window_size, center=True).mean()
        dfs.append(df)

    peaks_list = []
    for df in dfs:
        peaks, _ = find_peaks(df['rolling'], prominence=var_prominence)
        peaks_list.append(peaks)

    # 🔹 Print overzicht van pieken
    print(f"\n=== Piekoverzicht voor {naam} (subdir: {subdir}) ===")
    total_peaks = 0
    for nm, peaks in zip(names, peaks_list):
        count = len(peaks)
        total_peaks += count
        print(f"{nm:<20} → {count:>3} pieken")
    print(f"Totaal geteld: {total_peaks} pieken\n")

    # Alleen plotten als check en show_plots True zijn
    if check and show_plots:
        fig, axes = plt.subplots(len(dfs), 1, figsize=(8, 12), sharex=True)
        for ax, df, nm, peaks in zip(axes, dfs, names, peaks_list):
            ax.plot(df['X'], df['CH1'], label='Ruwe data', alpha=0.5)
            ax.plot(df['X'], df['rolling'], label='Rolling gemiddelde', color='orange')
            ax.plot(df['X'].iloc[peaks], df['rolling'].iloc[peaks], 'rx', label='Gevonden pieken')

            # Nummer bij elke piek zetten
            for j, p in enumerate(peaks):
                ax.text(df['X'].iloc[p], df['rolling'].iloc[p] + 0.02, str(j+1),
                        color='red', fontsize=8, ha='center')

            ax.set_title(f"{nm} (subdir: {subdir})")
            ax.set_ylabel('CH1 (intensiteit)')
            ax.legend()
            ax.grid(True)
        axes[-1].set_xlabel('X (hoek of tijd)')
        plt.tight_layout()
        plt.show()

    # Tel cumulatief de fringe-tellingen op
    fringes_0 = 0
    fringes_cum = [fringes_0]
    for peaks in peaks_list:
        fringes_cum.append(fringes_cum[-1] + len(peaks))

    return np.array(fringes_cum)


def fit_multidata(hoek_graden, N):
    popt, pcov = curve_fit(N_fit_suiker, hoek_graden, N, p0=[1.3])
    return popt

def plot_datagraphs(naam, x, y, n_line, n_fit):
    plt.figure(naam)
    plt.scatter(x, y, label='Measured Data', color='red')
    plt.plot(theta_fit, n_line, label=f'Fitted Curve (n={n_fit[0]:.2f})', color='blue')
    plt.plot(theta_fit, N_fit_suiker(theta_fit, 1.35), label='n = 1.35', color='orange')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('N')
    plt.title('Refractive Index Fit')
    plt.xlim(0, 12)
    plt.ylim(0, 200)
    plt.legend()
    plt.grid()

# -------------------------------
# Hoofdprogramma
# -------------------------------

theta_graden = np.array([0, 2, 4, 6, 8, 10])
theta_fit = np.linspace(0, 12, 100)
C = np.array([0, 0.25, 0.5, 0.75, 1.0])
C_cor = concentratie_correctie(V_totaal, C)
n_theorie = lin(f_sacharose, C_cor, n_water_532)

# Zet show_plots=True bij de reeks die je wilt bekijken
fringes_025ml = fringes('0.25ml', win_size, prom, subdir='0.25', show_plots=False)
fringes_05ml  = fringes('0.5ml',  win_size, prom, subdir='0.5',  show_plots=True)   # ← pieken zichtbaar
fringes_075ml = fringes('0.75ml', win_size, prom, subdir='0.75', show_plots=False)
fringes_1ml   = fringes('1ml',    win_size, prom, subdir='1',    show_plots=False)
fringes_blanco = fringes('blanco', win_size, prom, subdir='blanco', show_plots=False)

fit_025ml = fit_multidata(theta_graden, fringes_025ml)
fit_05ml  = fit_multidata(theta_graden, fringes_05ml)
fit_075ml = fit_multidata(theta_graden, fringes_075ml)
fit_1ml   = fit_multidata(theta_graden, fringes_1ml)
fit_blanco = fit_multidata(theta_graden, fringes_blanco)

fit_lijn_025ml = N_fit_suiker(theta_fit, *fit_025ml)
fit_lijn_05ml  = N_fit_suiker(theta_fit, *fit_05ml)
fit_lijn_075ml = N_fit_suiker(theta_fit, *fit_075ml)
fit_lijn_1ml   = N_fit_suiker(theta_fit, *fit_1ml)
fit_lijn_blanco = N_fit_suiker(theta_fit, *fit_blanco)

n = np.array([float(fit_blanco[0]), float(fit_025ml[0]), float(fit_05ml[0]),
              float(fit_075ml[0]), float(fit_1ml[0])])

# Lineaire fit over alle data
popt, pcov = curve_fit(lin, C_cor, n)
slope, intercept = popt
trend = lin(slope, C_cor, intercept)
trend_alt = lin(slope, C_cor, n_water_532)

# Samenvattende plot
plt.figure(0)
plt.scatter(C_cor, n, label='data', color='blue')
plt.plot(C_cor, trend, label=f'y = {round(slope,4)}x + {round(intercept,4)}', color='red', linestyle='--')
plt.plot(C_cor, n_theorie, label='waarde theoretisch', color='orange')
plt.plot(C_cor, trend_alt, label='alternatieve lijn', color='purple', linestyle='--')
plt.xlabel('m/v%')
plt.ylabel('n')
plt.legend()
plt.title('Totaaldata refractieve index')
plt.grid()

# Detailplots alleen als check=True
if check:
    plot_datagraphs('0.25ml', theta_graden, fringes_025ml, fit_lijn_025ml, fit_025ml)
    plot_datagraphs('0.5ml', theta_graden, fringes_05ml, fit_lijn_05ml, fit_05ml)
    plot_datagraphs('0.75ml', theta_graden, fringes_075ml, fit_lijn_075ml, fit_075ml)
    plot_datagraphs('1ml', theta_graden, fringes_1ml, fit_lijn_1ml, fit_1ml)
    plot_datagraphs('blanco', theta_graden, fringes_blanco, fit_lijn_blanco, fit_blanco)

print('C(m/v%)',C_cor)
print('n',n)
print('n_theorie',n_theorie)

plt.show()
