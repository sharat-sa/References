import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

#Disclaimer: The writing of this code was done by Sharat SA while assisted by LLMs in Google Antigravity IDE. LLMs were used to translate manually written pseudocode to Python, and for debugging. The core logic is entirely human.

# --- User Parameters ---
CUTOFF_FREQ  = 13e6   # Max measured frequency (Hz, 1e6 is 1MHz)
PPD          = 16     # Points per decade for extension, keep same as measurement
FIT_LB_START = 3e6   # Lower bound on fitted data, defined as earliest point where grain-boundary decays and grain response dominates
FIT_LB_STEP  = 0.5e6   # Step size when curve is not smooth at join (Hz) - Mismatch between measured and extrapolated curves comes when significant gb contributions still occurs even above FIT_LB_START frequency, we keep on raising till only grain response is captured and extrapolate
SMOOTH_TOL = 0.25   #10 perc defined as ad-hoc error tolerance for slope based on what results in low error

def analyze(path):
    p = Path(path)
    print(f"Processing: {p.name}")

    with open(p) as f_in:
        skip = next(i for i, l in enumerate(f_in) if 'Freq' in l)

    df = pd.read_csv(p, skiprows=skip, index_col=False)
    df.columns = df.columns.str.strip()

    f = df['Frequency (Hz)'].values
    z = df['Impedance Magnitude (Ohms)'].values * np.exp(1j * np.deg2rad(df["Impedance Phase Degrees (')"].values))

    if f.max() < 0.8 * CUTOFF_FREQ:
        print(f"  Skipped: max frequency {f.max()/1e6:.1f} MHz < {0.8*CUTOFF_FREQ/1e6:.1f} MHz (80% of CUTOFF_FREQ)")
        return

    f, z = f[f <= CUTOFF_FREQ], z[f <= CUTOFF_FREQ]
    w = 2 * np.pi * f
    f_max = f.max()

    def rc_model(w_in, L_val, R_val, C_val):
        Z = 1j * w_in * L_val + R_val / (1 + 1j * w_in * R_val * C_val)
        return np.r_[Z.real, Z.imag]

    # Dynamic lower-bound: raise fit_lb until dZ/df is continuous at the join
    fit_lb, L, R, C, z_corr = FIT_LB_START, None, None, None, None
    while fit_lb < 0.5*(CUTOFF_FREQ-FIT_LB_START):
        fm = f > fit_lb
        try:
            (L, R, C), _ = curve_fit(
                lambda x, L, R, C: rc_model(w[fm], L, R, C), f[fm],
                np.r_[z[fm].real, z[fm].imag], p0=[1e-7, 300, 1e-11])
            z_corr = z - 1j * w * L
            (R, C), _ = curve_fit(
                lambda x, R, C: rc_model(w[fm], 0, R, C), f[fm],
                np.r_[z_corr[fm].real, z_corr[fm].imag], p0=[R, C])
        except RuntimeError:
            fit_lb += FIT_LB_STEP; continue

        # Smoothness: relative error between exp and model dZ/df at join
        dZdf_exp = (z_corr[-1] - z_corr[-2]) / (f[-1] - f[-2])
        w_j = 2 * np.pi * f_max
        dZdf_mod = -1j * R**2 * C / (1 + 1j * w_j * R * C)**2 * (2 * np.pi)
        err = max(abs((dZdf_exp - dZdf_mod).real) / max(abs(dZdf_mod.real), 1e-12),
                  abs((dZdf_exp - dZdf_mod).imag) / max(abs(dZdf_mod.imag), 1e-12))

        print(f"  [lb={fit_lb/1e6:.1f} MHz] L={L:.2e} R={R:.1f} C={C:.2e} err={err:.2f}")
        if err <= SMOOTH_TOL:
            break
        fit_lb += FIT_LB_STEP

    # Extrapolate RC model from f_max to 10 GHz
    f_ex = np.logspace(np.log10(f_max), 10, int((10 - np.log10(f_max)) * PPD) + 1)[1:]
    z_ex = R / (1 + 1j * (2 * np.pi * f_ex) * R * C)

    df_out = pd.concat([
        pd.DataFrame({'f': f,    "z'": z_corr.real, "z''": z_corr.imag}),
        pd.DataFrame({'f': f_ex, "z'": z_ex.real,   "z''": z_ex.imag}),
    ]).sort_values('f').drop_duplicates('f').reset_index(drop=True)
    df_out.to_csv(p.with_suffix('.txt'), sep='\t', index=False, header=False)

    plt.figure()
    plt.plot(z_corr.real, -z_corr.imag, 'o', label='Exp (L-corrected)')
    plt.plot(z_ex.real,   -z_ex.imag,   '-', label='Sim (Extrapolated)')
    plt.xlabel("Z' (Ohms)"); plt.ylabel("-Z'' (Ohms)")
    plt.legend(); plt.grid()
    plt.savefig(p.with_suffix('.png')); plt.close()

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    for file in filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")]):
        analyze(file)