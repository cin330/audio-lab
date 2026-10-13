# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


class Recorder:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.stream = None
        self.parts = []

    def receive_audio(self, indata, frames, time_info, status):
        self.parts.append(indata[:, 0].copy())

    def start(self):
        if self.stream is not None:
            return
        self.parts = []
        self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                     dtype='float32', callback=self.receive_audio)
        try:
            self.stream.start()
        except Exception:
            self.stream.close()
            self.stream = None
            raise

    def stop(self):
        if self.stream is None:
            return np.array([], dtype=np.float32)
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.parts:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.parts)


class Player:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.audio = None
        self.position = 0.0
        self.speed = 1.0
        self.playing = False
        self.start_time = 0.0

    def load(self, audio):
        self.stop()
        self.audio = audio

    def duration(self):
        if self.audio is None:
            return 0.0
        return len(self.audio) / self.sample_rate

    def current_time(self):
        if not self.playing:
            return self.position
        elapsed = time.monotonic() - self.start_time
        position = min(self.duration(), self.position + elapsed * self.speed)
        if position >= self.duration():
            self.playing = False
            self.position = position
        return position

    def play(self):
        if self.audio is None or len(self.audio) == 0:
            raise ValueError('没有可播放的音频')
        if self.playing:
            return
        if self.position >= self.duration():
            self.position = 0.0
        start = int(self.position * self.sample_rate)
        part = self.audio[start:]
        if self.speed != 1.0:
            indices = np.arange(0, len(part), self.speed)
            part = np.interp(indices, np.arange(len(part)), part).astype('float32')
        sd.play(np.clip(part, -1, 1), self.sample_rate, blocking=False)
        self.start_time = time.monotonic()
        self.playing = True

    def pause(self):
        if self.playing:
            self.position = self.current_time()
        self.playing = False
        sd.stop()

    def stop(self):
        self.playing = False
        self.position = 0.0
        sd.stop()

    def seek(self, seconds):
        was_playing = self.playing
        self.pause()
        self.position = max(0, min(float(seconds), self.duration()))
        if was_playing and self.position < self.duration():
            self.play()

    def set_speed(self, speed):
        was_playing = self.playing
        self.pause()
        self.speed = float(speed)
        if was_playing:
            self.play()


def read_audio(path, target_rate):
    data, rate = sf.read(path, dtype='float32', always_2d=True)
    data = data.mean(axis=1)
    if rate != target_rate:
        factor = gcd(int(rate), int(target_rate))
        data = resample_poly(data, target_rate // factor, rate // factor)
    return data.astype('float32')


def write_audio(path, samples, sample_rate):
    sf.write(path, samples, sample_rate, subtype='FLOAT')
