# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:01:08 2024

@author: arjun
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def plot_CHG_spectra(D2, D3):
    
    D2 = D2 * 1e-30
    D3 = D3 * 1e-45
    
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
    
    plt.figure()
    plt.plot(ff, Ampf, label="Ampf")
    
    #D2 = 5000e-30
    #D3 = -30000e-45
    Phi = 1 / 2 * D2 * (ww - w0)**2 + D3 * (ww - w0)**3
    plt.plot(ff, Phi / np.max(Phi), label="Phi / max(Phi)")
    plt.legend()
    plt.show()
    
    tt = np.arange(-600e-15, 600e-15, 0.005e-15)
    pul = np.zeros(len(tt))
    for i in range(len(ff)):
        pul += Ampf[i] * np.sin(2 * np.pi * ff[i] * tt + Phi[i])
    
    pul /= np.max(pul)
    plt.figure()
    plt.plot(tt * 1e15, pul)
    plt.show()
    
    # Finding peaks
    peaks, _ = find_peaks(pul**2)
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    params, _ = curve_fit(gaussian, tt[peaks] * 1e15, pul[peaks]**2, p0=[1, 0, 10])
    gaussian_fit_sigma = params[2]
    print("Gaussian fit FWHM:", gaussian_fit_sigma / np.sqrt(2) * 2.355)
    
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
    
    spec_l = np.array(spec_l)
    
    plt.figure()
    plt.imshow(spec_l, aspect='auto', extent=[7.1379e14, 7.892e14, 0, 130], origin='lower', cmap='viridis')
    plt.colorbar(label="Spectral Intensity")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("r56 (micrometers)")
    plt.show()

#%%

def plot_CHG_spectra_slice(D2, D3, R56):
    
    D2 = D2 * 1e-30
    D3 = D3 * 1e-45
    R56 = R56 * 1e-6
    
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
    
    #D2 = 5000e-30
    #D3 = -30000e-45
    Phi = 1 / 2 * D2 * (ww - w0)**2 + D3 * (ww - w0)**3
    
    tt = np.arange(-600e-15, 600e-15, 0.005e-15)
    pul = np.zeros(len(tt))
    for i in range(len(ff)):
        pul += Ampf[i] * np.sin(2 * np.pi * ff[i] * tt + Phi[i])
    
    pul /= np.max(pul)
   
    # Finding peaks
    peaks, _ = find_peaks(pul**2)
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    params, _ = curve_fit(gaussian, tt[peaks] * 1e15, pul[peaks]**2, p0=[1, 0, 10])
    gaussian_fit_sigma = params[2]
    print("Gaussian fit FWHM:", gaussian_fit_sigma / np.sqrt(2) * 2.355)
    
    zz0 = tt * c0
    dE = np.random.normal(0, sigmae, len(zz0))
    dE += pul * Emod


    zz = zz0 + dE * R56
    dens, _ = np.histogram(zz, bins=16001)
    spec = np.abs(np.fft.fftshift(np.fft.fft(dens)))**2

    T = (np.max(tt) - np.min(tt)) / len(dens)
    Fs = 1 / T
    L = len(dens)
    dF = Fs / L
    freq = np.linspace(-Fs / 2, Fs / 2 - dF, L)

    f2i = int(np.floor((7.1379e14 - np.min(freq)) / dF))
    f2l = int(np.floor((7.892e14 - np.min(freq)) / dF))
        
    plt.plot(c0/freq[f2i:f2l] * 1e9, spec[f2i:f2l])
    
    return freq[f2i:f2l], spec[f2i:f2l]


#%%
def calc_bn(tau0, wl, printmax = True):
    wl = np.asarray(wl).reshape(-1,)
    bn = np.zeros(len(wl))
    for i in range(len(wl)):
        z = np.sum(np.exp(-1j * 2 * np.pi * (tau0 / wl[i])))
        bn[i] = abs(z) / len(tau0)
    
    index = np.argmax(bn)
    wl_max = wl[index]
    if printmax:
        print("Maximum bunching factor is", np.round(max(bn),4) , " at " , np.round(wl_max*1e9,2) , " nm")
    return(np.array(bn))


def plot_slice(z, wl, slice_len=0, n_slice=40):
    if slice_len != 0:
        n_slice = int((max(z) - min(z)) / slice_len)

    zz = np.linspace(min(z), max(z), n_slice)
    bn, z_slice = [], []
    
    for i in range(1,len(zz)):
        z1, z2 = zz[i - 1], zz[i]
        z_slice.append(np.mean([z1, z2]))
        slice_zz = z[(z >= z1) * (z < z2)]
        #print(len(slice_zz))
        if len(slice_zz) == 0:
            bn.append(0)
        else:
            bn.append(max(calc_bn(slice_zz, wl, printmax = False)))
        i += 1
    
    z_slice = np.array(z_slice) - np.mean(z_slice)
    bn      = np.array(bn)
    
    plt.figure()
    plt.plot(z_slice, bn)
    plt.xlabel('s (m)')
    plt.ylabel('Bunching Factor')
    return(z_slice, bn)