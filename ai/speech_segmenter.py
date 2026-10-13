# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

"""Collect complete speech phrases using the bundled Silero VAD."""

import numpy as np


class SpeechSegmenter:
    def __init__(self, sample_rate=16000, normalizer=None, detector=None,
                 max_seconds=12, probe_seconds=.4):
        self.sample_rate = sample_rate
        self.normalizer = normalizer or (lambda audio: audio)
        self.detector = detector or self.detect
        self.max_samples = int(max_seconds * sample_rate)
        self.probe_samples = int(probe_seconds * sample_rate)
        self.pending = np.empty(0, dtype='float32')
        self.new_samples = 0

    @staticmethod
    def detect(audio):
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        return get_speech_timestamps(audio, VadOptions(
            threshold=.5, min_speech_duration_ms=250,
            min_silence_duration_ms=650, speech_pad_ms=250,
        ))

    def reset(self):
        self.pending = np.empty(0, dtype='float32')
        self.new_samples = 0

    def push(self, audio):
        self.pending = np.concatenate((self.pending, audio))
        self.new_samples += len(audio)
        if self.new_samples < self.probe_samples and len(self.pending) < self.max_samples:
            return []
        self.new_samples = 0
        prepared = self.normalizer(self.pending)
        chunks = self.detector(prepared) if prepared is not None else []
        if not chunks:
            # Keep one second of pre-roll so soft word onsets are not cut away.
            self.pending = self.pending[-self.sample_rate:]
            return []
        ready, consumed = [], 0
        for chunk in chunks:
            # An ongoing phrase ends at the current buffer edge. Completed
            # phrases leave a silence gap after their padded endpoint.
            if chunk['end'] <= len(self.pending) - int(.2 * self.sample_rate):
                ready.append(self.pending[chunk['start']:chunk['end']].copy())
                consumed = chunk['end']
        if consumed:
            self.pending = self.pending[consumed:]
        elif len(self.pending) >= self.max_samples:
            ready.append(self.pending[chunks[0]['start']:self.max_samples].copy())
            self.pending = self.pending[self.max_samples:]
        return ready

    def flush(self):
        prepared = self.normalizer(self.pending) if len(self.pending) else None
        chunks = self.detector(prepared) if prepared is not None else []
        ready = [self.pending[chunk['start']:chunk['end']].copy() for chunk in chunks]
        self.reset()
        return ready
