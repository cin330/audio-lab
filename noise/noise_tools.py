# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np


def gaussian_noise(audio, snr):
    audio = np.asarray(audio, dtype='float32')
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-12:
        raise ValueError('原始音频太安静，无法按目标 SNR 加噪')
    noise = np.random.normal(size=len(audio))
    target = signal_power / (10 ** (snr / 10))
    noise = noise * np.sqrt(target / np.mean(noise ** 2))
    noise = noise.astype('float32')
    return audio + noise, noise


def impulse_noise(audio, probability):
    audio = np.asarray(audio, dtype='float32')
    noise = np.zeros(len(audio), dtype='float32')
    mask = np.random.random(len(audio)) < probability / 100
    rms = np.sqrt(np.mean(audio ** 2))
    noise[mask] = 3 * max(rms, 0.01) * np.random.choice([-1, 1], mask.sum())
    return audio + noise, noise
