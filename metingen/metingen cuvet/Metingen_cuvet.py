import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Pad aanpassen aan jouw situatie
#main_path = 'C:\\Users\\luukj\\Documents\\GitHub\\Project-2.1\\metingen\\metingen cuvet'
main_path = 'C:\\Users\\Win11 Pro\\Documents\\Project-2.1\\metingen\\metingen cuvet'
os.chdir(main_path)

# constants
lambda_laser = 532e-9  # wavelength of the laser in meters
d_cuvett = (1.25e-2 - 1e-2 )/2
n_air = 1.0 
thickness = d_cuvett

win_size = 10 
prom = 0.5

# Zet op True om piekplots te tonen
check = True

def lin(a, x, b):
    return a * x + b

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

def N_fit(theta_in,n):
    """Function to fit the refractive index data."""
    theta_in_rad = np.deg2rad(theta_in)
    return (2*thickness/lambda_laser) * (np.sqrt(n**2 - np.sin(theta_in_rad)**2) - np.cos(theta_in_rad)+1-n)

def fringes(naam, var_window_size, var_prominence, subdir, show_plots=False, x_ranges=None):
    """
    Bepaalt fringes, toont optioneel piekdetectieplots en print samenvatting.
    Je kunt optioneel per bestand een X-bereik instellen via een dict 'x_ranges':
        {'0.5ml_0g2g': (0.1, 0.8), '0.5ml_2g4g': (0.2, 1.0), ...}
    Alleen pieken BINNEN dat X-bereik worden geteld.
    """
    #naam_1 = f"{naam}_0g2g"
    naam_2 = f"{naam}_2g4g"
    naam_3 = f"{naam}_4g6g"
    naam_4 = f"{naam}_6g8g"
    naam_5 = f"{naam}_8g10g"
    #names = [naam_1, naam_2, naam_3, naam_4, naam_5]
    names = [naam_2, naam_3, naam_4, naam_5]

    dfs, peaks_list = [], []

    print(f"\n=== Piekoverzicht voor {naam} (subdir: {subdir}) ===")
    total_peaks = 0

    for nm in names:
        df = read(nm, subdir)
        df['rolling'] = df['CH1'].rolling(window=var_window_size, center=True).mean()

        # Bepaal X-bereik
        if x_ranges and nm in x_ranges:
            x_min, x_max = x_ranges[nm]
            mask = (df['X'] >= x_min) & (df['X'] <= x_max)
            df_window = df[mask]
        else:
            df_window = df
            x_min, x_max = df['X'].min(), df['X'].max()

        # Piekdetectie alleen binnen geselecteerd X-bereik
        peaks, _ = find_peaks(df_window['rolling'], prominence=var_prominence)

        # Opslaan voor later gebruik
        dfs.append((df, df_window, peaks, (x_min, x_max)))
        total_peaks += len(peaks)
        print(f"{nm:<20} → {len(peaks):>3} pieken (X bereik [{x_min:.2f}, {x_max:.2f}])")

    print(f"Totaal geteld: {total_peaks} pieken\n")

    # Plotten (alleen bij check=True en show_plots=True)
    if check and show_plots:
        fig, axes = plt.subplots(len(dfs), 1, figsize=(8, 12), sharex=True)
        for ax, (df, df_window, peaks, (x_min, x_max)) in zip(axes, dfs):
            ax.plot(df['X'], df['CH1'], alpha=0.4, label='Ruwe data')
            ax.plot(df['X'], df['rolling'], color='orange', label='Rolling gemiddelde')

            # Highlight X-bereik visueel
            ax.axvspan(x_min, x_max, color='green', alpha=0.1, label='Gebruikt X-bereik')

            # Pieken binnen bereik tonen
            if len(peaks) > 0:
                ax.plot(df_window['X'].iloc[peaks], df_window['rolling'].iloc[peaks],
                        'rx', label='Gevonden pieken')
                for j, p in enumerate(peaks):
                    ax.text(df_window['X'].iloc[p], df_window['rolling'].iloc[p] + 0.02,
                            str(j + 1), color='red', fontsize=8, ha='center')

            ax.set_title(f"{df_window.index.name or naam}")
            ax.legend()
            ax.grid(True)

        axes[-1].set_xlabel("Hoek (°)")
        plt.tight_layout()
        #plt.show()

    # Cumulatieve fringes berekenen
    fringes_0 = 0
    fringes_cum = [fringes_0]
    for _, _, peaks, _ in dfs:
        fringes_cum.append(fringes_cum[-1] + len(peaks))

    return np.array(fringes_cum)


def fit_multidata(hoek_graden, N, naam='', check=False):
    """
    Fit de fringedata aan het model en retourneert de best-fit n_suiker.
    Houdt rekening met relatieve verschuivingen (ΔN).
    """
    N = np.array(N, dtype=float)
    N -= N[0]  # relatieve fringes

    def model_shifted(theta, n_suiker):
        vals = N_fit(theta, n_suiker)
        return vals - vals[0]

    # Fit uitvoeren
    popt, pcov = curve_fit(model_shifted, hoek_graden, N, p0=[1.33])
    n_fit = popt[0]
    return popt

def plot_datagraphs(naam, x, y, n_line, n_fit):
    plt.figure(naam)
    plt.scatter(x, y, label='Measured Data', color='red')
    plt.plot(theta_fit, n_line, label=f'Fitted Curve (n={n_fit[0]:.2f})', color='blue')
    plt.plot(theta_fit, N_fit(theta_fit, 1.33), label='n = 1.33', color='orange')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('N')
    plt.title('Refractive Index Fit')
    plt.xlim(0, 12)
    plt.ylim(0, 250)
    plt.legend()
    plt.grid()

#theta_graden = np.array([0, 2 ,4, 6, 8, 10])
theta_graden = np.array([0, 4, 6, 8, 10])
theta_fit = np.linspace(0, 12, 100)

# Zet show_plots=True bij de reeks die je wilt bekijken
x_ranges_1 = {
    '1_0g2g': (106, 630),
    '1_2g4g': (230, 740),
    '1_4g6g': (237, 724),
    '1_6g8g': (337, 820),
    '1_8g10g': (128, 1040)
}
x_ranges_2 = {
    '2_0g2g': (262, 707),
    '2_2g4g': (353, 892),
    '2_4g6g': (486, 1002),
    '2_6g8g': (420, 895),
    '2_8g10g': (270, 853)
}
x_ranges_3 = {
    '3_0g2g': (32, 379),
    '3_2g4g': (282, 883),
    '3_4g6g': (568, 919),
    '3_6g8g': (306, 767),
    '3_8g10g': (207, 743)
}
x_ranges_4 = {
    '4_0g2g': (141, 383),
    '4_2g4g': (395, 750),
    '4_4g6g': (195, 610),
    '4_6g8g': (159, 794),
    '4_8g10g': (135, 721)
}
x_ranges_5 = {
    '5_0g2g': (158, 614),
    '5_2g4g': (261, 727),
    '5_4g6g': (213, 694),
    '5_6g8g': (262, 644),
    '5_8g10g': (239, 784)
}

fringes_1 = fringes(
    '1', (win_size), prom,
    subdir='1', show_plots=True,
    x_ranges=x_ranges_1
)
fringes_2 = fringes(
    '2', win_size, prom,
    subdir='2', show_plots=True,
    x_ranges=x_ranges_2
)
fringes_3 = fringes(
    '3', win_size, prom,
    subdir='3', show_plots=True,
    x_ranges=x_ranges_3
)
fringes_4 = fringes(
    '4', (win_size), prom,
    subdir='4', show_plots=True,
    x_ranges=x_ranges_4
)
fringes_5 = fringes(
    '5', (win_size), prom,
    subdir='5', show_plots=True,
    x_ranges=x_ranges_5
)

fit_1 = fit_multidata(theta_graden, fringes_1)
fit_2  = fit_multidata(theta_graden, fringes_2)
fit_3 = fit_multidata(theta_graden, fringes_3)
fit_4   = fit_multidata(theta_graden, fringes_4)
fit_5 = fit_multidata(theta_graden, fringes_5)

fit_1 = fit_multidata(theta_graden, fringes_1, naam='1', check=check)
fit_2 = fit_multidata(theta_graden, fringes_2, naam='2', check=check)
fit_3 = fit_multidata(theta_graden, fringes_3, naam='3', check=check)
fit_4 = fit_multidata(theta_graden, fringes_4, naam='4', check=check)
fit_5 = fit_multidata(theta_graden, fringes_5, naam='5', check=check)

fit_lijn_1 = N_fit(theta_fit, *fit_1)
fit_lijn_2  = N_fit(theta_fit, *fit_2)
fit_lijn_3 = N_fit(theta_fit, *fit_3)
fit_lijn_4   = N_fit(theta_fit, *fit_4)
fit_lijn_5 = N_fit(theta_fit, *fit_5)


n = np.array([float(fit_1[0]), float(fit_2[0]), float(fit_3[0]),
              float(fit_4[0]), float(fit_5[0])])
aantal = np.arange(0,len(n))
# Samenvattende plot
plt.figure(0)
plt.scatter(aantal, n, label='data', color='blue')
plt.plot(aantal, n, label='data', color='blue',linestyle='--')

plt.xlabel('#')
plt.ylabel('n')
plt.legend()
plt.title('Totaaldata refractieve index')
plt.grid()

# Detailplots alleen als check=True
if check:
    plot_datagraphs('1', theta_graden, fringes_1, fit_lijn_1, fit_1)
    plot_datagraphs('2', theta_graden, fringes_2, fit_lijn_2, fit_2)
    plot_datagraphs('3', theta_graden, fringes_3, fit_lijn_3, fit_3)
    plot_datagraphs('4', theta_graden, fringes_4, fit_lijn_4, fit_4)
    plot_datagraphs('5', theta_graden, fringes_5, fit_lijn_5, fit_5)

n_gem = np.mean(n)
stdev = np.std(n)
print('n gemiddeld: ', round(n_gem,2), '+/-', round(stdev,2))
print('fout is: ', round(stdev/np.sqrt(len(n)),2))

plt.show()