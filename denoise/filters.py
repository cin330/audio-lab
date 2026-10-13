# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np
import pywt
from scipy import signal
from scipy.ndimage import uniform_filter


def wiener_filter(audio, fs, noise=None):
    size = min(1024, len(audio))
    _, _, stft = signal.stft(audio, fs, nperseg=size, noverlap=size // 2)
    power = abs(stft) ** 2
    if noise is None:
        noise_power = np.percentile(power, 20, axis=1, keepdims=True)
    else:
        _, _, noise_stft = signal.stft(noise, fs, nperseg=size, noverlap=size // 2)
        noise_power = np.mean(abs(noise_stft) ** 2, axis=1, keepdims=True)
    gain = np.clip(1 - noise_power / (power + 1e-12), 0.06, 1)
    gain = uniform_filter(gain, size=(3, 3))
    _, output = signal.istft(stft * gain, fs, nperseg=size, noverlap=size // 2)
    return output[:len(audio)].astype('float32')


def kalman_filter(audio, fs, noise=None):
    # 简化的一阶 AR 模型卡尔曼滤波，适合教学演示。
    values = np.asarray(audio, dtype='float64')
    average = values.mean()
    values = values - average
    power = max(np.mean(values ** 2), 1e-10)
    if noise is None:
        block = min(256, max(32, len(values) // 8))
        chunks = values[:len(values) // block * block].reshape(-1, block)
        r = np.percentile(np.mean(chunks ** 2, axis=1), 15) * 0.7
    else:
        r = np.mean(np.asarray(noise) ** 2)
    r = float(np.clip(r, power * 0.001, power * 0.9))
    denominator = max(np.dot(values[:-1], values[:-1]) - len(values) * r, 1e-9)
    a = float(np.clip(np.dot(values[1:], values[:-1]) / denominator, -0.98, 0.98))
    q = max(power * 0.04, r * 0.15, 1e-9)
    output = np.empty_like(values)
    state = 0.0
    variance = power
    for i in range(len(values)):
        prediction = a * state
        predicted_variance = a * a * variance + q
        gain = predicted_variance / (predicted_variance + r)
        state = prediction + gain * (values[i] - prediction)
        variance = (1 - gain) * predicted_variance
        output[i] = state + average
    return output.astype('float32')


def wavelet_filter(audio, fs, noise=None):
    wavelet = 'db4'
    level = min(5, pywt.dwt_max_level(len(audio), pywt.Wavelet(wavelet).dec_len))
    if level < 1:
        return np.array(audio, dtype='float32')
    coefficients = pywt.wavedec(audio, wavelet, level=level)
    detail = coefficients[-1]
    if noise is not None:
        detail = pywt.wavedec(noise, wavelet, level=level)[-1]
    sigma = np.median(abs(detail)) / 0.6745
    threshold = 0.8 * sigma * np.sqrt(2 * np.log(len(audio)))
    new_coefficients = [coefficients[0]]
    for part in coefficients[1:]:
        new_coefficients.append(pywt.threshold(part, threshold, mode='soft'))
    output = pywt.waverec(new_coefficients, wavelet)
    return output[:len(audio)].astype('float32')


def run_filter(name, audio, fs, noise=None):
    if name == '维纳滤波':
        return wiener_filter(audio, fs, noise)
    if name == '卡尔曼滤波':
        return kalman_filter(audio, fs, noise)
    if name == '小波去噪':
        return wavelet_filter(audio, fs, noise)
    raise ValueError('不存在的算法：' + name)
