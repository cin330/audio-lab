# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import unittest

import numpy as np

from ai.speech_segmenter import SpeechSegmenter


def detector(audio):
    """Deterministic speech marks for testing endpoint handling, not VAD accuracy."""
    indices = np.flatnonzero(audio)
    if not len(indices):
        return []
    return [{'start': int(indices[0]), 'end': int(indices[-1] + 1)}]


class SegmentationTests(unittest.TestCase):
    def test_phrase_is_not_cut_at_three_seconds(self):
        segmenter = SpeechSegmenter(sample_rate=100, detector=detector)
        for _ in range(50):
            self.assertEqual(segmenter.push(np.ones(10)), [])
        ready = []
        for _ in range(4):
            ready.extend(segmenter.push(np.zeros(10)))
        self.assertEqual(len(ready), 1)
        self.assertEqual(len(ready[0]), 500)

    def test_continuous_speech_is_bounded(self):
        segmenter = SpeechSegmenter(sample_rate=100, detector=detector)
        ready = []
        for _ in range(121):
            ready.extend(segmenter.push(np.ones(10)))
        self.assertEqual(len(ready), 1)
        self.assertEqual(len(ready[0]), 1200)
        self.assertEqual(len(segmenter.pending), 10)

    def test_stop_flush_preserves_short_last_phrase(self):
        segmenter = SpeechSegmenter(sample_rate=100, detector=detector)
        for _ in range(9):
            segmenter.push(np.ones(10))
        ready = segmenter.flush()
        self.assertEqual(len(ready), 1)
        self.assertEqual(len(ready[0]), 90)
        self.assertFalse(len(segmenter.pending))

    def test_idle_silence_has_bounded_storage(self):
        segmenter = SpeechSegmenter(sample_rate=100, detector=detector)
        for _ in range(100):
            self.assertEqual(segmenter.push(np.zeros(10)), [])
        self.assertLessEqual(len(segmenter.pending), 140)
        self.assertEqual(segmenter.flush(), [])


if __name__ == '__main__':
    unittest.main()
