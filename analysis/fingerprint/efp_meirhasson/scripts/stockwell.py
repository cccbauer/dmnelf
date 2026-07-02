#!/usr/bin/env python3
"""
stockwell.py
------------
FFT-based discrete Stockwell (S-) transform, as used by Meir-Hasson et al. 2014
for the EEG Finger-Print time/frequency representation (1 Hz resolution).

The S-transform of a length-N signal h at frequency f (in samples^-1, n = f*N*dt)
is computed efficiently in the Fourier domain (Stockwell et al. 1996):

    S(τ, n) = IFFT_m[ H(m + n) * G(m, n) ]

where H = FFT(h) and G(m, n) = exp(-2 π² m² / n²) is the Gaussian window whose
width scales with frequency (narrower at higher frequency). We compute S only at
the integer Hz frequencies we need (1..fmax), which keeps memory modest.

Reference: Stockwell, Mansinha & Lowe (1996), IEEE Trans. Signal Process. 44(4).
"""
import numpy as np


def _gaussian(n_len, freq_bin):
    """Frequency-domain Gaussian window for S-transform at integer freq_bin (>0)."""
    m = np.arange(n_len)
    # wrap m to [-N/2, N/2) so the Gaussian is centered correctly for circular FFT
    m = np.where(m > n_len // 2, m - n_len, m)
    return np.exp(-2.0 * (np.pi ** 2) * (m ** 2) / (freq_bin ** 2))


def stockwell_power(x, sfreq, fmin=1, fmax=40):
    """
    Power S-transform of a 1-D signal at integer Hz frequencies fmin..fmax.

    Parameters
    ----------
    x : (n_samples,) array
    sfreq : float, sampling frequency (Hz)
    fmin, fmax : int, integer frequency range in Hz (1 Hz steps)

    Returns
    -------
    freqs : (n_freq,) integer Hz
    power : (n_freq, n_samples) |S|^2
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    N = x.shape[0]
    H = np.fft.fft(x)
    freqs = np.arange(int(fmin), int(fmax) + 1)
    power = np.empty((len(freqs), N), dtype=np.float64)

    for i, f in enumerate(freqs):
        # integer frequency bin corresponding to f Hz
        n = int(round(f * N / sfreq))
        if n <= 0:
            power[i] = 0.0
            continue
        # shifted spectrum H(m+n), circular
        Hshift = np.roll(H, -n)
        S = np.fft.ifft(Hshift * _gaussian(N, n))
        power[i] = np.abs(S) ** 2

    return freqs, power


def _selftest():
    """Sanity check: a pure tone should peak at its frequency; a chirp should track."""
    sf = 250.0
    t = np.arange(0, 8, 1 / sf)
    # 10 Hz tone
    tone = np.sin(2 * np.pi * 10 * t)
    f, p = stockwell_power(tone, sf, 1, 40)
    peak = f[np.argmax(p.mean(axis=1))]
    print(f"[tone] dominant frequency = {peak} Hz (expected 10)")
    assert peak == 10, "tone peak mismatch"

    # linear chirp 5 -> 25 Hz
    inst = 5 + (25 - 5) * (t / t[-1])
    phase = 2 * np.pi * np.cumsum(inst) / sf
    chirp = np.sin(phase)
    f, p = stockwell_power(chirp, sf, 1, 40)
    # dominant frequency early vs late should increase
    early = f[np.argmax(p[:, : len(t) // 4].mean(axis=1))]
    late = f[np.argmax(p[:, -len(t) // 4 :].mean(axis=1))]
    print(f"[chirp] early≈{early} Hz, late≈{late} Hz (expected increasing 5->25)")
    assert late > early, "chirp does not track upward"
    print("stockwell self-test passed.")


if __name__ == "__main__":
    _selftest()
