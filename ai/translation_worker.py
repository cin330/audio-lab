# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

"""Microphone/file speech recognition and English translation on a Qt worker."""

import queue
import re
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QThread, Signal

from ai.cuda_runtime import prepare_cuda_libraries
from ai.speech_segmenter import SpeechSegmenter


LANGUAGE_CODES = {'中文': 'zh', '英语': 'en', '日语': 'ja', '韩语': 'ko',
                  '法语': 'fr', '德语': 'de', '西班牙语': 'es'}


class TranslationWorker(QThread):
    result_signal = Signal(str, str)
    status_signal = Signal(str)
    error_signal = Signal(str)
    audio_signal = Signal(object, int)

    def __init__(self, language='zh', audio_path=None, parent=None, model_size='medium'):
        super().__init__(parent)
        self.language = language
        self.audio_path = Path(audio_path) if audio_path else None
        if model_size not in ['base', 'medium']:
            raise ValueError('不支持的模型：' + model_size)
        self.model_size = model_size
        self.sample_rate = 16000
        self.chunk_seconds = 12
        self.block_size = 1600
        self.audio_queue = queue.Queue(maxsize=300)
        self.running = False
        self._stop_event = threading.Event()
        self._finish_event = threading.Event()
        self._microphone_active = threading.Event()
        self._overflow_event = threading.Event()
        self.previous_text = {'transcribe': '', 'translate': ''}
        self.error = ''

    def stopped(self):
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def audio_callback(self, indata, frames, time_info, status):
        if self.stopped() or self._finish_event.is_set() or not self.running:
            return
        audio = indata[:, 0].copy()
        self.audio_signal.emit(audio, self.sample_rate)
        try:
            self.audio_queue.put_nowait(audio)
        except queue.Full:
            self._overflow_event.set()
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(audio)
            except queue.Full:
                pass

    def create_model(self):
        prepare_cuda_libraries()
        # Optional AI dependencies are imported only when translation is started.
        from faster_whisper import WhisperModel
        folder = Path(__file__).resolve().parent.parent / 'data' / 'models' / f'whisper-{self.model_size}'
        model_path = str(folder) if (folder / 'model.bin').is_file() else self.model_size
        try:
            return WhisperModel(model_path, device='cuda', compute_type='int8_float16',
                                local_files_only=True)
        except Exception as error:
            raise RuntimeError(f'{self.model_size} 模型加载失败：{error}') from error

    def recognize(self, model, audio, task):
        options = dict(language=self.language, task=task, beam_size=5, temperature=0,
                       vad_filter=True, condition_on_previous_text=True,
                       vad_parameters=dict(min_silence_duration_ms=650, speech_pad_ms=250))
        prompt = self.previous_text[task]
        if self.language == 'zh' and task == 'transcribe':
            prompt = '以下是普通话的句子。' + prompt
        if prompt:
            options['initial_prompt'] = prompt
        text = self.decode(model, audio, options)
        if self.stopped():
            return ''
        if task == 'translate' and re.search(r'[\u3400-\u9fff]', text):
            # Retry a mixed-language translation without the previous prompt,
            # rather than carrying source-language leakage into the next turn.
            retry_options = dict(options, condition_on_previous_text=False)
            retry_options.pop('initial_prompt', None)
            retry = self.decode(model, audio, retry_options)
            if self.stopped():
                return ''
            if retry and not re.search(r'[\u3400-\u9fff]', retry):
                text = retry
        if text:
            self.previous_text[task] = ('' if task == 'translate' and re.search(r'[\u3400-\u9fff]', text)
                                        else text[-300:])
        return text

    def decode(self, model, audio, options):
        segments, _ = model.transcribe(audio, **options)
        parts = []
        try:
            for segment in segments:
                if self.stopped():
                    return ''
                parts.append(segment.text)
        finally:
            close = getattr(segments, 'close', None)
            if close:
                close()
        return ''.join(parts).strip()

    def translate_audio(self, model, audio):
        self.status_signal.emit('识别中')
        original = self.recognize(model, audio, 'transcribe')
        if self.stopped():
            return False
        if not original:
            return False
        # Display the source text before the second inference completes.
        self.result_signal.emit(original, '')
        if self.stopped():
            return True
        self.status_signal.emit('翻译中')
        translated = original if self.language == 'en' else self.recognize(model, audio, 'translate')
        if not self.stopped() and translated:
            self.result_signal.emit('', translated)
        return True

    @staticmethod
    def prepare_microphone_audio(audio):
        audio = np.asarray(audio, dtype='float32')
        if not len(audio) or not np.all(np.isfinite(audio)):
            return None
        rms = float(np.sqrt(np.mean(audio.astype('float64') ** 2)))
        if rms < 1e-5:
            return None
        peak = float(np.max(np.abs(audio)))
        # Quiet speech should reach VAD rather than be discarded by a fixed gate.
        gain = min(20.0, max(1.0, 0.03 / rms), 0.95 / peak)
        return (audio * gain).astype('float32')

    def run_microphone(self, model):
        segmenter = SpeechSegmenter(self.sample_rate, self.prepare_microphone_audio,
                                    max_seconds=self.chunk_seconds)
        def process(audio):
            prepared = self.prepare_microphone_audio(audio)
            if prepared is not None and not self.stopped():
                recognized = self.translate_audio(model, prepared)
                if not self.stopped():
                    self.status_signal.emit('监听中' if recognized else '未检测到语音')
        # Announce listening only after the device has opened successfully.
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32',
                            blocksize=self.block_size, callback=self.audio_callback):
            self._microphone_active.set()
            self.status_signal.emit('监听中')
            try:
                while not self.stopped():
                    try:
                        part = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        if self._finish_event.is_set():
                            break
                        continue
                    if self._overflow_event.is_set():
                        segmenter.reset()
                        self.previous_text = {'transcribe': '', 'translate': ''}
                        self._overflow_event.clear()
                        self.status_signal.emit('已跳过积压音频')
                    for phrase in segmenter.push(part):
                        process(phrase)
                        if self.stopped():
                            break
                if self._finish_event.is_set() and not self.stopped():
                    for phrase in segmenter.flush():
                        process(phrase)
            finally:
                self._microphone_active.clear()

    def run(self):
        # A stop requested just after start() must survive thread initialization.
        if self.stopped():
            return
        self.running = True
        try:
            audio = None
            if self.audio_path is not None:
                if not self.audio_path.is_file():
                    raise FileNotFoundError(f'找不到音频文件：{self.audio_path}')
                from faster_whisper.audio import decode_audio
                self.status_signal.emit('读取音频')
                audio = decode_audio(str(self.audio_path), sampling_rate=self.sample_rate)
                if not len(audio):
                    raise ValueError('音频文件没有有效采样点')
                if self.stopped():
                    return
                self.audio_signal.emit(audio, self.sample_rate)
            self.status_signal.emit(f'加载 {self.model_size}')
            model = self.create_model()
            if self.stopped():
                return
            if audio is not None:
                self.translate_audio(model, audio)
            else:
                self.run_microphone(model)
        except Exception as error:
            self.error = str(error)
            self.error_signal.emit(self.error)
        finally:
            self.running = False
            if not self.error:
                self.status_signal.emit('已停止' if self.stopped() or self.audio_path is None else '完成')

    def stop(self, finish_pending=False):
        if finish_pending and self._microphone_active.is_set():
            self._finish_event.set()
            return
        self._stop_event.set()
        self.running = False
        self.requestInterruption()
