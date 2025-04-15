# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:04:11 2024

@author: arjun
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.widgets import Slider

# Constants
c0 = 2.9979e8
sigmae = 0.0007
Emod = 0.0060
t0 = 45e-15
bwd = np.sqrt(2) * 0.441 / t0
sigf = bwd / 2.355
wl0 = 800e-9
f0 = c0 / wl0
w0 = 2 * np.pi * f0

fi = f0 + 50e12
fe = f0 - 50e12
ff = np.linspace(fi, fe, 401)
ww = 2 * np.pi * ff
Ampf = np.exp(-(ff - f0)**2 / (2 * sigf**2))

def calculate_spec_l(D2, D3):
    Phi = 1 / 2 * D2*1e-30 * (ww - w0)**2 + D3*1e-45 * (ww - w0)**3

    tt = np.arange(-600e-15, 600e-15, 0.005e-15)
    pul = np.zeros(len(tt))
    for i in range(len(ff)):
        pul += Ampf[i] * np.sin(2 * np.pi * ff[i] * tt + Phi[i])

    pul /= np.max(pul)

    zz0 = tt * c0
    dE = np.random.normal(0, sigmae, len(zz0))
    dE += pul * Emod

    r56_l = np.linspace(0, 130e-6, 131)
    spec_l = []
    for r56 in r56_l:
        zz = zz0 + dE * r56
        dens, _ = np.histogram(zz, bins=16001)
        spec = np.abs(np.fft.fftshift(np.fft.fft(dens)))**2

        T = (np.max(tt) - np.min(tt)) / len(dens)
        Fs = 1 / T
        L = len(dens)
        dF = Fs / L
        freq = np.linspace(-Fs / 2, Fs / 2 - dF, L)

        f2i = int(np.floor((7.1379e14 - np.min(freq)) / dF))
        f2l = int(np.floor((7.892e14 - np.min(freq)) / dF))
        spec_l.append(spec[f2i:f2l])

    return np.array(spec_l)

def update(val):
    D2 = slider_D2.val
    D3 = slider_D3.val
    spec_l = calculate_spec_l(D2, D3)
    
    ax1.imshow(spec_l, aspect='auto', extent=[7.1379e14, 7.892e14, 0, 130], origin='lower', cmap='viridis')
    ax2.clear()
    ax2.plot(np.linspace(7.1379e14, 7.892e14, spec_l.shape[1]), spec_l[60])
    ax2.set_title(f"Spectrum at r56[80] (D2={D2:.1e}, D3={D3:.1e})")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Spectral Intensity")
    fig.canvas.draw_idle()

# Initial values for D2 and D3
D2 = 5000
D3 = -30000

# Initial calculation
spec_l = calculate_spec_l(D2, D3)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.25, bottom=0.25)

# First plot (interactive spectrum visualization)
ax1.imshow(spec_l, aspect='auto', extent=[7.1379e14, 7.892e14, 0, 130], origin='lower', cmap='viridis')
ax1.set_title("Spectral Intensity vs Frequency and r56")
ax1.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("r56 (micrometers)")
ax1.colorbar = fig.colorbar(ax1.images[0], ax=ax1, label="Spectral Intensity")

# Second plot (specific slice visualization)
ax2.plot(np.linspace(7.1379e14, 7.892e14, spec_l.shape[1]), spec_l[60])
ax2.set_title(f"Spectrum at r56[80] (D2={D2:.1e}, D3={D3:.1e})")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Spectral Intensity")

# Add sliders for D2 and D3
ax_D2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
ax_D3 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')

slider_D2 = Slider(ax_D2, 'D2', 0, 10000, valinit=D2, valstep=100e-30)
slider_D3 = Slider(ax_D3, 'D3', -50000, 50000, valinit=D3, valstep=1000e-45)

slider_D2.on_changed(update)
slider_D3.on_changed(update)

plt.show()
