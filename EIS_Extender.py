import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

#Disclaimer: The writing of this code was done by Sharat SA with LLM assisstance for debugging and traslating pseudocode to python. Logic is entirely human generated. 

# --- User Parameters ---
# Might need changing based on grain/grain-boundary conductivity in samples and measurement conditions.  
CUTOFF_FREQ  = 13e6   # Max reliable measured frequency (Hz, 1e6 is 1MHz)
PPD          = 16     # Points per decade for extension, keep same as measurement
FIT_LB_START = 3e6   # Lower bound on fitted data, defined as earliest point where grain-boundary decays and grain response dominates
FIT_LB_STEP  = 0.5e6   # Step size when curve is not smooth at join (Hz) - Mismatch between measured and extrapolated curves comes when significant gb contributions still occurs even above FIT_LB_START frequency, we keep on raising till only grain response is captured and extrapolate
SMOOTH_TOL     = 0.25   #25 perc defined as ad-hoc error tolerance for slope based on what results in low error
MIN_FREQ_FRAC  = 0.5  # Skip file if max frequency < this fraction of CUTOFF_FREQ

def analyze(path):
    #File input and filtering
    p = Path(path)
    print(f"Processing: {p.name}")

    with open(p) as f_in:
        skip = next(i for i, l in enumerate(f_in) if 'Freq' in l)

    df = pd.read_csv(p, skiprows=skip).rename(columns=str.strip)
    df = df.dropna(subset=['Frequency (Hz)', 'Impedance Magnitude (Ohms)', "Impedance Phase Degrees (')"])

    f = df['Frequency (Hz)'].to_numpy(float)
    mag = df['Impedance Magnitude (Ohms)'].to_numpy(float)
    ph = df["Impedance Phase Degrees (')"].to_numpy(float)
    z = mag * np.exp(1j * np.radians(ph)) #Use complx  directly, we'll separate real and imaginary while saving

    if f.max() < MIN_FREQ_FRAC * CUTOFF_FREQ:
        reason = f"max frequency {f.max()/1e6:.2f} MHz < {MIN_FREQ_FRAC*CUTOFF_FREQ/1e6:.2f} MHz ({MIN_FREQ_FRAC:.0%} of CUTOFF_FREQ)"
        print(f"  Skipped: {reason}")
        return reason

    #Initialise and sort
    m = f <= CUTOFF_FREQ
    idx = np.argsort(f[m])
    f, z = f[m][idx], z[m][idx]
    w = 2 * np.pi * f

    def rc_model(w_in, L, R, C):
        Z = 1j * w_in * L + R / (1 + 1j * w_in * R * C)
        return np.r_[Z.real, Z.imag]

    # Fit and extrapolate HF data; while loop increases the lower bound till error minimises
    fit_lb, L, R, C, z_corr = FIT_LB_START, None, None, None, None
    while fit_lb < CUTOFF_FREQ:
        fm = f > fit_lb
        if fm.sum() < 3: #Fit doesn't work if number of data points <3 
            break
        try:
            (L, R, C), _ = curve_fit(rc_model, w[fm], np.r_[z[fm].real, z[fm].imag], p0=[1e-7, 300, 1e-11])
            z_corr = z - 1j * w * L #for the sake of smoothness, we go over entire set
        except (RuntimeError, TypeError, ValueError):
            fit_lb += FIT_LB_STEP; continue

        # Smoothness: relative error between exp and model dZ/df at join
        dZdf_exp = (z_corr[-1] - z_corr[-2]) / (f[-1] - f[-2]) #tangent at last two points (join) of measured data
        dZdf_mod = -1j * R**2 * C / (1 + 1j * w[-1] * R * C)**2 * (2 * np.pi) #tangent at joint, measured from fitted RC
        diff = dZdf_exp - dZdf_mod
        err = max(abs(diff.real) / max(abs(dZdf_mod.real), 1e-12), abs(diff.imag) / max(abs(dZdf_mod.imag), 1e-12)) #error can be from either real or imaginary part, we consider max

        print(f"  [lb={fit_lb/1e6:.1f} MHz] L={L:.2e} R={R:.1f} C={C:.2e} err={err:.2f}")
        if err <= SMOOTH_TOL:
            break
        fit_lb += FIT_LB_STEP

    if z_corr is None:
        reason = "could not fit data"
        print(f"  Skipped: {reason} for {p.name}")
        return reason

    # Extrapolate RC model from f_max to 10 GHz 
    f_ex = np.logspace(np.log10(f[-1]), 10, int((10 - np.log10(f[-1])) * PPD) + 1)[1:]
    w_ex = 2 * np.pi * f_ex
    z_ex = R / (1 + 1j * w_ex * R * C)

    #Save txt and png
    f_all, z_all = np.r_[f, f_ex], np.r_[z_corr, z_ex]
    np.savetxt(p.with_suffix('.txt'), np.column_stack([f_all, z_all.real, z_all.imag]), delimiter='\t')

    plt.figure()
    plt.plot(z_corr.real, -z_corr.imag, 'o', label='Exp (L-corrected)')
    plt.plot(z_ex.real,   -z_ex.imag,   '-', label='Sim (Extrapolated)')
    plt.xlabel("Z' (Ohms)"); plt.ylabel("-Z'' (Ohms)")
    plt.legend(); plt.grid()
    plt.savefig(p.with_suffix('.png')); plt.close()
    return None

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    files = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
    skipped = [(f, reason) for f in files if (reason := analyze(f))]
    if skipped:
        print(f"\nUnprocessed files ({len(skipped)}/{len(files)}):")
        for f, reason in skipped:
            print(f"  - {Path(f).name}: {reason}")
    elif files:
        print(f"\nAll {len(files)} file(s) processed successfully.")