# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import numpy as np
from scipy.signal import butter, sosfiltfilt
from denoise.filters import wiener_filter


def change_tone(audio, fs, bass_db=0, treble_db=0):
    if len(audio) < 128:
        return audio.copy()
    low_cut = min(350, fs * 0.2)
    high_cut = min(3500, fs * 0.35)
    low_filter = butter(2, low_cut, btype='low', fs=fs, output='sos')
    high_filter = butter(2, high_cut, btype='high', fs=fs, output='sos')
    bass = sosfiltfilt(low_filter, audio)
    treble = sosfiltfilt(high_filter, audio)
    output = audio + (10 ** (bass_db / 20) - 1) * bass
    output = output + (10 ** (treble_db / 20) - 1) * treble
    return output.astype('float32')


def process_scene(name, audio, fs, noise=None, strength=1.0):
    if not 0 <= strength <= 1:
        raise ValueError('增强强度必须在 0 到 1 之间')
    if name not in ['突出人声', '突出音乐', '突出环境', '原始平衡']:
        raise ValueError('未知场景：' + name)
    if name == '原始平衡' or strength == 0:
        return audio.copy()
    if name == '突出人声':
        clean = wiener_filter(audio, fs, noise)
        processed = change_tone(clean, fs, bass_db=-1.0, treble_db=1.5)
    elif name == '突出音乐':
        processed = change_tone(audio, fs, bass_db=0.4, treble_db=2.0)
    else:
        processed = change_tone(audio, fs, bass_db=0.5, treble_db=0.7)
    if strength == 1:
        return processed
    return ((1 - strength) * audio + strength * processed).astype('float32')
