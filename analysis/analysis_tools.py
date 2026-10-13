# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np
from scipy.fft import rfft, rfftfreq


def waveform(audio, fs):
    if audio is None:
        return [], []
    step = max(1, len(audio) // 6000)
    return np.arange(0, len(audio), step) / fs, audio[::step]


def fft_spectrum(audio, fs):
    n = min(32768, len(audio))
    if n < 8:
        return [], []
    part = audio[:n]
    window = np.hanning(n)
    magnitude = 2 * np.abs(rfft(part * window)) / max(window.sum(), 1)
    return rfftfreq(n, 1 / fs), 20 * np.log10(np.maximum(magnitude, 1e-8))


def calculate_snr(clean, processed):
    if clean is None or processed is None:
        return None
    signal_power = np.mean(np.asarray(clean, dtype='float64') ** 2)
    error_power = np.mean((np.asarray(clean) - np.asarray(processed)) ** 2)
    if signal_power < 1e-12:
        return None
    if error_power < 1e-15:
        return float('inf')
    return float(10 * np.log10(signal_power / error_power))


def snr_text(value):
    if value is None:
        return '无参考'
    return f'{value:.2f} dB'
