import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

main_path = 'C:\\Users\\luukj\\Documents\\GitHub\\Project-2.1\\metingen\\metingen suiker'
os.chdir(main_path)

# constants
lambda_laser = 532e-9 # wavelength of the laser in meters
d_sugar= 1e-2 # thickness in meters
d_cuvett = 1.25e-2 - d_sugar # thickness in meters
n_cuvett = 1.59 # refractive index of the cuvett
n_air = 1.0023 # refractive index of air

n_water_532 = 1.3382
f_sacharose = 1.77e-3
V_totaal = 3

win_size = 5
prom = 0.25

check = False

def lin(a,x,b):
    return a*x+b
def concentratie_correctie(V,C):
    return (C/3)*100 

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

def N_fit_suiker(theta_in,n_sugar):
    """Function to fit the refractive index data."""
    theta_in_rad = np.deg2rad(theta_in)
    return 2/lambda_laser*(2*d_cuvett*(np.sqrt(n_cuvett**2 -n_air**2*np.sin(theta_in_rad)**2)-n_air*np.cos(theta_in_rad)) + d_sugar*(np.sqrt(n_sugar**2 -n_air**2*np.sin(theta_in_rad)**2)-n_air*np.cos(theta_in_rad))-(2*n_cuvett*d_cuvett + n_sugar*d_sugar - 2*d_cuvett*n_air-d_sugar*n_air))

def fringes(naam,var_window_size,var_prominence,subdir):
    naam_1 = str(naam) + '_0g2g'
    naam_2 = str(naam) + '_2g4g'
    naam_3 = str(naam) + '_4g6g'
    naam_4 = str(naam) + '_6g8g'
    naam_5 = str(naam) + '_8g10g'

    df02 = read(naam_1,subdir)
    df24 = read(naam_2,subdir)
    df46 = read(naam_3,subdir)
    df68 = read(naam_4,subdir)
    df810 = read(naam_5,subdir)

    # Rolling averages toevoegen
    df02['rolling'] = df02['CH1'].rolling(window=var_window_size, center=True).mean()
    df24['rolling'] = df24['CH1'].rolling(window=var_window_size, center=True).mean()
    df46['rolling'] = df46['CH1'].rolling(window=var_window_size, center=True).mean()
    df68['rolling'] = df68['CH1'].rolling(window=var_window_size, center=True).mean()
    df810['rolling'] = df810['CH1'].rolling(window=var_window_size, center=True).mean()

    peaks_02, props_02 = find_peaks(df02['rolling'], prominence=var_prominence)
    peaks_24, props_24 = find_peaks(df24['rolling'], prominence=var_prominence)
    peaks_46, props_46 = find_peaks(df46['rolling'], prominence=var_prominence)
    peaks_68, props_68 = find_peaks(df68['rolling'], prominence=var_prominence)
    peaks_810, props_810 = find_peaks(df810['rolling'], prominence=var_prominence)

    fringes_0 = 0
    fringes_02 = fringes_0 + len(peaks_02)
    fringes_24 = fringes_02 + len(peaks_24)
    fringes_46 = fringes_24 + len(peaks_46)
    fringes_68 = fringes_46 + len(peaks_68)
    fringes_810 = fringes_68 + len(peaks_810)
    fringes = np.array([fringes_0,fringes_02,fringes_24,fringes_46,fringes_68,fringes_810])

    return fringes
def fit_multidata(hoek_graden,N):
    popt, pcov = curve_fit(N_fit_suiker, hoek_graden, N , p0=[1.3])
    return popt
def plot_datagraphs(naam,x,y,n_line,n_fit):
    # plotting
    plt.figure(naam)
    plt.scatter(x, y, label='Measured Data', color='red')
    plt.plot(theta_fit, n_line, label=f'Fitted Curve (n={n_fit[0]:.2f})', color='blue')
    plt.plot(theta_fit,N_fit_suiker(theta_fit,1.35),label='n = 1.35',color='orange')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('N')
    plt.title('Refractive Index Fit')
    plt.xlim(0,12)
    plt.ylim(0,200)          
    plt.legend()
    plt.grid()

#vast
theta_graden = np.array([0,2,4,6,8,10])
theta_fit = np.linspace(0,12,100)
C = np.array([0,0.25,0.5,0.75,1.0])
C_cor = concentratie_correctie(V_totaal,C)
n_theorie = lin(f_sacharose,C_cor,n_water_532)

fringes_025ml = fringes('0.25ml',win_size,(prom+0.05),subdir='0.25')
fit_025ml = fit_multidata(theta_graden,fringes_025ml)
fit_lijn_025ml = N_fit_suiker(theta_fit, * fit_025ml)

fringes_05ml = fringes('0.5ml',win_size,prom,subdir='0.5')
fit_05ml = fit_multidata(theta_graden,fringes_05ml)
fit_lijn_05ml = N_fit_suiker(theta_fit, * fit_05ml)

fringes_075ml = fringes('0.75ml',win_size,(prom-1),subdir='0.75')
fit_075ml = fit_multidata(theta_graden,fringes_05ml)
fit_lijn_075ml = N_fit_suiker(theta_fit, * fit_075ml)

fringes_1ml = fringes('1ml',win_size,(prom),subdir='1')
fit_1ml = fit_multidata(theta_graden,fringes_1ml)
fit_lijn_1ml = N_fit_suiker(theta_fit, * fit_1ml)

fringes_blanco = fringes('blanco',win_size,prom,subdir='blanco')
fit_blanco = fit_multidata(theta_graden,fringes_blanco)
fit_lijn_blanco = N_fit_suiker(theta_fit, * fit_blanco)

n = np.array([float(fit_blanco),float(fit_025ml),float(fit_05ml),float(fit_075ml),float(fit_1ml)])

#C_cor = np.delete(C_cor,(len(C_cor)-1))
#n = np.delete(n,(len(n)-1))
#n_theorie = lin(f_sacharose,C_cor,n_water_532)

popt,pcov = curve_fit(lin,C_cor,n,)
slope,intercept = popt
trend = lin(slope,C_cor,intercept)
trend_alt = lin(slope,C_cor,n_water_532)

plt.figure(0)
plt.scatter(C_cor,n,label='data',color='blue')
plt.plot(C_cor,trend,label=f'y = {round(slope,4)}x + {round(intercept,4)}',color='red',linestyle='--')
plt.plot(C_cor,n_theorie,label='waarde theoretisch',color='orange')
plt.plot(C_cor,trend_alt,label='alternatieve lijn',color='purple',linestyle='--')
plt.xlabel('m/v%')
plt.ylabel('n')
plt.legend()
plt.title('totaal data')
plt.grid()

if check == True:
    plot_datagraphs('0.25ml',theta_graden,fringes_025ml,fit_lijn_025ml,fit_025ml)
    plot_datagraphs('0.5ml',theta_graden,fringes_05ml,fit_lijn_05ml,fit_05ml)
    plot_datagraphs('0.75ml',theta_graden,fringes_075ml,fit_lijn_075ml,fit_075ml)
    plot_datagraphs('1ml',theta_graden,fringes_1ml,fit_lijn_1ml,fit_1ml)
    plot_datagraphs('blanco',theta_graden,fringes_blanco,fit_lijn_blanco,fit_blanco)

plt.show()