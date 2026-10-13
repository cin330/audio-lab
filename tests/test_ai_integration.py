# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

"""Worker cancellation, scene strength and GUI signal integration checks."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from ai.translation_worker import TranslationWorker
from ai.speech_segmenter import SpeechSegmenter
from enhancement.scene_modes import process_scene
from gui.main_window import AudioLabWindow

APP = QApplication.instance() or QApplication([])


def pump_until(condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        APP.processEvents()
        time.sleep(.005)
    APP.processEvents()
    if not condition():
        raise AssertionError('Qt operation timed out')


class WorkerTests(unittest.TestCase):
    def test_stop_before_run_does_not_load_model_or_open_microphone(self):
        worker = TranslationWorker()
        worker.stop()
        with patch.object(worker, 'create_model') as model, patch('ai.translation_worker.sd.InputStream') as stream:
            worker.run()
            model.assert_not_called()
            stream.assert_not_called()

    def test_stop_during_model_loading_does_not_open_microphone(self):
        entered, release = threading.Event(), threading.Event()
        worker = TranslationWorker()
        def load():
            entered.set()
            release.wait(2)
            return Mock()
        with patch.object(worker, 'create_model', side_effect=load), patch('ai.translation_worker.sd.InputStream') as stream:
            worker.start()
            self.assertTrue(entered.wait(1))
            worker.stop()
            release.set()
            self.assertTrue(worker.wait(2000))
            stream.assert_not_called()

    def test_queue_overflow_keeps_recent_audio(self):
        worker = TranslationWorker()
        worker.running = True
        for index in range(310):
            worker.audio_callback(np.full((8, 1), index, dtype='float32'), 8, None, None)
        self.assertEqual(worker.audio_queue.qsize(), 300)
        self.assertEqual(worker.audio_queue.get()[0], 10)

    def test_language_and_prompt_are_used_only_for_transcription(self):
        worker = TranslationWorker(language='zh')
        model = Mock()
        model.transcribe.side_effect = [
            (iter([SimpleNamespace(text='你好')]), None),
            (iter([SimpleNamespace(text='Hello')]), None),
        ]
        results = []
        worker.result_signal.connect(lambda source, target: results.append((source, target)))
        worker.translate_audio(model, np.ones(16000, dtype='float32'))
        self.assertEqual(results, [('你好', ''), ('', 'Hello')])
        self.assertIn('initial_prompt', model.transcribe.call_args_list[0].kwargs)
        self.assertNotIn('initial_prompt', model.transcribe.call_args_list[1].kwargs)
        self.assertEqual(model.transcribe.call_args_list[1].kwargs['task'], 'translate')
        self.assertEqual(model.transcribe.call_args_list[0].kwargs['beam_size'], 5)
        model.transcribe.side_effect = [(iter([SimpleNamespace(text='后一句')]), None)]
        worker.recognize(model, np.ones(16000, dtype='float32'), 'transcribe')
        self.assertIn('你好', model.transcribe.call_args.kwargs['initial_prompt'])

    def test_error_is_not_overwritten_by_stopped_status(self):
        worker = TranslationWorker()
        errors, statuses = [], []
        worker.error_signal.connect(errors.append)
        worker.status_signal.connect(statuses.append)
        with patch.object(worker, 'create_model', side_effect=RuntimeError('GPU unavailable')):
            worker.run()
        self.assertEqual(errors, ['GPU unavailable'])
        self.assertNotIn('已停止', statuses)
        self.assertFalse(worker.running)

    def test_mixed_language_translation_is_retried_without_context(self):
        worker = TranslationWorker()
        worker.previous_text['translate'] = 'Previous sentence.'
        model = Mock()
        model.transcribe.side_effect = [
            (iter([SimpleNamespace(text='中文 English')]), None),
            (iter([SimpleNamespace(text='English translation')]), None),
        ]
        text = worker.recognize(model, np.ones(16000, dtype='float32'), 'translate')
        self.assertEqual(text, 'English translation')
        self.assertNotIn('initial_prompt', model.transcribe.call_args.kwargs)
        self.assertFalse(model.transcribe.call_args.kwargs['condition_on_previous_text'])
        self.assertEqual(worker.previous_text['translate'], 'English translation')

    def test_microphone_chunks_produce_results_and_release_stream_on_stop(self):
        worker = TranslationWorker()
        model = Mock()
        model.transcribe.side_effect = [
            (iter([SimpleNamespace(text='原文')]), None),
            (iter([SimpleNamespace(text='Translation')]), None),
        ]
        released, results = [], []

        @contextmanager
        def stream(**options):
            try:
                for _ in range(30):
                    options['callback'](np.full((1600, 1), .1, dtype='float32'), 1600, None, None)
                for _ in range(8):
                    options['callback'](np.zeros((1600, 1), dtype='float32'), 1600, None, None)
                yield
            finally:
                released.append(True)

        def received(source, target):
            results.append((source, target))
            if target:
                worker.stop()

        worker.result_signal.connect(received)
        detector = lambda audio: [{'start': 0, 'end': min(len(audio), 48000)}]
        real_segmenter = SpeechSegmenter(normalizer=worker.prepare_microphone_audio, detector=detector)
        with patch.object(worker, 'create_model', return_value=model), patch('ai.translation_worker.sd.InputStream', side_effect=stream), patch('ai.translation_worker.SpeechSegmenter', return_value=real_segmenter):
            worker.run()
        self.assertEqual(results, [('原文', ''), ('', 'Translation')])
        self.assertEqual(released, [True])
        self.assertFalse(worker.running)

    def test_quiet_speech_is_prepared_instead_of_discarded(self):
        t = np.arange(48000) / 16000
        quiet = (.002 * np.sin(2 * np.pi * 220 * t)).astype('float32')
        self.assertLess(np.sqrt(np.mean(quiet ** 2)), .005)
        prepared = TranslationWorker.prepare_microphone_audio(quiet)
        self.assertIsNotNone(prepared)
        self.assertGreater(np.sqrt(np.mean(prepared ** 2)), .005)
        self.assertLessEqual(np.max(np.abs(prepared)), .95)
        self.assertIsNone(TranslationWorker.prepare_microphone_audio(np.zeros(48000)))

    def test_finish_pending_translates_short_final_phrase(self):
        worker = TranslationWorker()
        model = Mock()
        model.transcribe.side_effect = [
            (iter([SimpleNamespace(text='最后一句')]), None),
            (iter([SimpleNamespace(text='The last sentence')]), None),
        ]
        results = []
        worker.result_signal.connect(lambda source, target: results.append((source, target)))
        worker.status_signal.connect(lambda status: worker.stop(finish_pending=True)
                                     if status == '监听中' else None)

        @contextmanager
        def stream(**options):
            for _ in range(18):
                options['callback'](np.full((1600, 1), .1, dtype='float32'), 1600, None, None)
            yield

        segmenter = SpeechSegmenter(normalizer=worker.prepare_microphone_audio,
                                    detector=lambda audio: [{'start': 0, 'end': len(audio)}])
        with patch.object(worker, 'create_model', return_value=model), patch('ai.translation_worker.sd.InputStream', side_effect=stream), patch('ai.translation_worker.SpeechSegmenter', return_value=segmenter):
            worker.run()
        self.assertEqual(results, [('最后一句', ''), ('', 'The last sentence')])
        self.assertFalse(worker.running)

    def test_no_speech_does_not_generate_a_translation(self):
        worker = TranslationWorker()
        model = Mock()
        model.transcribe.return_value = (iter([]), None)
        self.assertFalse(worker.translate_audio(model, np.zeros(48000, dtype='float32')))
        self.assertEqual(model.transcribe.call_count, 1)

    def test_strength_endpoints_and_interpolation(self):
        audio = np.random.default_rng(7).normal(0, .05, 4096).astype('float32')
        for mode in ['突出人声', '突出音乐', '突出环境', '原始平衡']:
            with self.subTest(mode=mode):
                full = process_scene(mode, audio, 16000, strength=1)
                self.assertTrue(np.array_equal(process_scene(mode, audio, 16000, strength=0), audio))
                half = process_scene(mode, audio, 16000, strength=.5)
                self.assertTrue(np.allclose(half, (audio + full) / 2))
        with self.assertRaises(ValueError):
            process_scene('突出音乐', audio, 16000, strength=70)


class FakeTranslationWorker(QThread):
    result_signal = Signal(str, str)
    status_signal = Signal(str)
    error_signal = Signal(str)
    audio_signal = Signal(object, int)

    def __init__(self, language='zh', audio_path=None, model_size='medium'):
        super().__init__()
        self.language = language
        self.audio_path = Path(audio_path) if audio_path else None
        self.model_size = model_size
        self.done = threading.Event()

    def run(self):
        self.status_signal.emit('监听中')
        self.audio_signal.emit(np.ones(1600, dtype='float32') * .05, 16000)
        self.result_signal.emit('测试原文', 'Test translation')
        self.done.wait(2)

    def stop(self, finish_pending=False):
        self.done.set()


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.window = AudioLabWindow()
        self.window.show()
        APP.processEvents()

    def tearDown(self):
        if self.window.translation_worker is not None:
            self.window.stop_translation()
            pump_until(lambda: self.window.translation_worker is None)
        self.window.close()
        APP.processEvents()

    def test_gui_results_waveform_gating_and_stop(self):
        w = self.window
        with patch('gui.main_window.TranslationWorker', FakeTranslationWorker):
            w.show_translation()
            w.translation_start_button.click()
            self.assertFalse(w.record_button.isEnabled())
            self.assertFalse(w.translation_import_button.isEnabled())
            pump_until(lambda: bool(w.translation_text.toPlainText()))
            self.assertEqual(w.asr_text.toPlainText(), '测试原文')
            self.assertEqual(w.translation_text.toPlainText(), 'Test translation')
            self.assertEqual(len(w.translation_plot.listDataItems()), 1)
            w.translation_start_button.click()
            pump_until(lambda: w.translation_worker is None)
            self.assertTrue(w.record_button.isEnabled())
            self.assertTrue(w.translation_import_button.isEnabled())
            self.assertEqual(w.translation_start_button.text(), '开始')

    def test_balance_uses_original_and_percentage_is_normalized(self):
        w = self.window
        audio = np.random.default_rng(9).normal(0, .05, 4096).astype('float32')
        w.set_audio(audio, 'test.wav')
        w.add_noise()
        w.show_scene()
        w.scene_input_combo.setCurrentText('加噪音频')
        w.scene_mode_buttons[3].click()
        self.assertEqual(w.scene_input_combo.currentText(), '原始音频')
        self.assertFalse(w.scene_input_combo.isEnabled())
        w.apply_scene()
        pump_until(lambda: w.worker is None)
        self.assertTrue(np.array_equal(w.scene_audio, w.original))
        self.assertTrue(w.scene_result_valid())
        self.assertEqual(w.scene_result_strength, 70)
        self.assertEqual(len(w.scene_fft_plot.listDataItems()), 2)

    def test_selected_quality_mode_is_passed_to_worker(self):
        w = self.window
        w.translation_model_combo.setCurrentIndex(1)
        with patch('gui.main_window.TranslationWorker', FakeTranslationWorker):
            w.start_translation()
            self.assertEqual(w.translation_worker.model_size, 'base')
            self.assertFalse(w.translation_model_combo.isEnabled())
            w.stop_translation()
            pump_until(lambda: w.translation_worker is None)
            self.assertTrue(w.translation_model_combo.isEnabled())

    def test_close_stops_translation_without_destroying_running_thread(self):
        w = self.window
        with patch('gui.main_window.TranslationWorker', FakeTranslationWorker), patch('gui.main_window.QMessageBox.question', return_value=QMessageBox.StandardButton.Yes):
            w.start_translation()
            w.close()
            pump_until(lambda: not w.isVisible())
            self.assertIsNone(w.translation_worker)


if __name__ == '__main__':
    unittest.main()
