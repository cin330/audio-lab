# SPDX-FileCopyrightText: 2026 cin330
# SPDX-License-Identifier: AGPL-3.0-only

import csv
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QThread, QTimer, Qt, Signal, QSize
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSlider, QFileDialog,
    QMessageBox, QStackedWidget, QFrame, QScrollArea, QProgressBar,
    QButtonGroup, QSizePolicy, QGridLayout, QToolButton, QCheckBox, QPlainTextEdit
)
from config import SAMPLE_RATE, METHODS
from audio.audio_tools import Recorder, Player, read_audio, write_audio
from noise.noise_tools import gaussian_noise, impulse_noise
from denoise.filters import run_filter
from analysis.analysis_tools import waveform, fft_spectrum, calculate_snr, snr_text
from enhancement.scene_modes import process_scene
from ai.translation_worker import TranslationWorker, LANGUAGE_CODES


# 耗时运算放在另一个线程，避免窗口卡住。
# 这里只是普通的类，没有使用装饰器。
class FilterThread(QThread):
    progress = Signal(str, int, int)

    def __init__(self, methods, audio, fs, noise, scene=False, strength=1.0):
        super().__init__()
        self.methods = list(methods)
        self.audio = audio.copy()
        self.fs = fs
        self.noise = None if noise is None else noise.copy()
        self.results = {}
        self.times = {}
        self.error = ''
        self.scene = scene
        self.strength = strength

    def run(self):
        try:
            for index, name in enumerate(self.methods):
                if self.isInterruptionRequested():
                    break
                self.progress.emit(name, index, len(self.methods))
                start = time.perf_counter()
                if self.scene:
                    self.results[name] = process_scene(name, self.audio, self.fs, self.noise,
                                                      strength=self.strength)
                else:
                    self.results[name] = run_filter(name, self.audio, self.fs, self.noise)
                self.times[name] = time.perf_counter() - start
                self.progress.emit(name, index + 1, len(self.methods))
        except Exception as error:
            self.error = str(error)


class AudioPlot(pg.PlotWidget):
    """Keep the empty-state hint inside the chart when the window is resized."""

    def __init__(self, title):
        super().__init__()
        self.setTitle(title, color='#34465e', size='12pt')
        self.empty_hint = QLabel('未加载音频', self)
        self.empty_hint.setObjectName('plotHint')
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'empty_hint'):
            self.empty_hint.setGeometry(65, 40, max(0, self.width() - 85),
                                        max(0, self.height() - 85))


class AudioLabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MYTU\'s AudioLab - 语音采集与降噪系统')
        self.resize(1440, 960)
        self.setMinimumSize(1080, 720)
        self.setFont(QFont('Noto Sans CJK SC', 10))

        self.fs = SAMPLE_RATE
        self.recorder = Recorder(self.fs)
        self.player = Player(self.fs)

        self.original = None
        self.noisy = None
        self.added_noise = None
        self.results = {}
        self.times = {}
        self.scene_audio = None
        self.scene_input_reference = None
        self.scene_result_mode = None
        self.scene_result_strength = None
        self.worker = None
        self.translation_worker = None
        self.translation_stopping = False
        self.translation_error = ''
        self.translation_audio = None
        self.translation_audio_fs = 16000
        self.close_after_translation = False
        self.record_started = 0.0
        self.previous_playing = False
        self.waveform_limit = 1.0

        self.build_ui()
        self.apply_style()
        self.refresh_graphs()
        self.refresh_scene_graphs()
        self.refresh_translation_graph()
        self.update_controls()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)

    def make_button(self, text, function, role='secondary', tip=None, icon=None):
        button = QPushButton(text)
        button.setProperty('role', role)
        button.setMinimumHeight(36)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(function)
        button.setToolTip(tip or text)
        button.setAccessibleName(tip or text)
        if icon:
            button.setIcon(self.icon(icon, white=role == 'primary'))
            button.setIconSize(QSize(18, 18))
        if not text:
            button.setProperty('iconOnly', True)
            button.setFixedSize(36, 36)
        return button

    def icon(self, name, white=False):
        suffix = '-white' if white else ''
        return QIcon(str(Path(__file__).parent / 'icons' / f'{name}{suffix}.svg'))

    def label(self, text, role='body'):
        label = QLabel(text)
        label.setProperty('textRole', role)
        label.setWordWrap(True)
        return label

    def card(self, title, step=None):
        frame = QFrame()
        frame.setObjectName('card')
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(9)
        if title:
            row = QHBoxLayout()
            if step:
                symbol = QLabel()
                symbol.setPixmap(self.icon({'01': 'noise', '02': 'sliders', '03': 'chart'}[step]).pixmap(18, 18))
                row.addWidget(symbol)
            row.addWidget(self.label(title, 'section'), 1)
            layout.addLayout(row)
        return frame, layout

    def scroll_area(self, content):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(content)
        return area

    def build_ui(self):
        root = QWidget()
        root.setObjectName('appRoot')
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setObjectName('header')
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 12, 24, 12)
        header_layout.setSpacing(24)
        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand.addWidget(self.label('AudioLab', 'brand'))
        header_layout.addLayout(brand)
        header_layout.addStretch()
        self.nav_group = QButtonGroup(self)
        self.nav_buttons = []
        for index, text in enumerate(['音频实验', '场景增强', '实时翻译']):
            button = self.make_button(text, lambda checked=False, i=index: self.switch_page(i), 'nav', icon=['waveform', 'sliders', 'languages'][index])
            button.setCheckable(True)
            self.nav_group.addButton(button, index)
            self.nav_buttons.append(button)
            header_layout.addWidget(button)
        self.nav_buttons[0].setChecked(True)
        header_layout.addStretch()
        self.state_badge = self.label('●  待导入', 'badge')
        self.state_badge.setMinimumWidth(105)
        self.state_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.state_badge)
        layout.addWidget(header)

        self.pages = QStackedWidget()
        self.pages.addWidget(self.make_experiment_page())
        self.pages.addWidget(self.make_scene_page())
        self.pages.addWidget(self.make_translation_page())
        layout.addWidget(self.pages, 1)
        self.statusBar().setSizeGripEnabled(False)
       

    def switch_page(self, index):
        changed = self.pages.currentIndex() != index
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        if changed:
            self.load_active_audio()

    def show_experiment(self):
        self.switch_page(0)

    def show_scene(self):
        self.switch_page(1)

    def show_translation(self):
        self.switch_page(2)

    def make_experiment_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(14)
        heading_row = QHBoxLayout()
        heading = QVBoxLayout()
        heading.setSpacing(3)
        heading.addWidget(self.label('音频实验', 'heading'))
        
        heading_row.addLayout(heading, 1)
        self.record_button = self.make_button('录音', self.toggle_record, 'primary', tip='开始 / 结束录音', icon='mic')
        self.import_button = self.make_button('导入', self.import_audio, tip='导入音频（WAV、FLAC、OGG、AIFF）', icon='upload')
        heading_row.addWidget(self.record_button)
        heading_row.addWidget(self.import_button)
        layout.addLayout(heading_row)

        info = QFrame()
        info.setObjectName('infoBar')
        info_row = QHBoxLayout(info)
        info_row.setContentsMargins(14, 9, 14, 9)
       
        self.audio_label = QLabel('未加载音频')
        self.audio_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.audio_meta = self.label('48 kHz', 'caption')
        info_row.addWidget(self.audio_label, 1)
        info_row.addWidget(self.audio_meta)
        layout.addWidget(info)

        body = QHBoxLayout()
        body.setSpacing(16)
        sidebar = QWidget()
        sidebar.setObjectName('sidebar')
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(12)

        noise_card, controls = self.card('添加噪声', '01')
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(['高斯噪声', '脉冲噪声'])
        self.noise_combo.currentTextChanged.connect(self.update_noise_text)
        controls.addWidget(self.noise_combo)
        self.noise_value_label = self.label('目标 SNR：10 dB', 'section')
        controls.addWidget(self.noise_value_label)
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 30)
        self.noise_slider.setValue(10)
        self.noise_slider.setAccessibleName('噪声强度')
        self.noise_slider.valueChanged.connect(self.update_noise_text)
        controls.addWidget(self.noise_slider)
        self.noise_slider.setToolTip('SNR 越低，噪声越强')
        self.noise_button = self.make_button('加噪', self.add_noise, 'soft', icon='noise')
        controls.addWidget(self.noise_button)
        sidebar_layout.addWidget(noise_card)

        method_card, controls = self.card('降噪处理', '02')
        self.method_combo = QComboBox()
        self.method_combo.addItems(METHODS)
        controls.addWidget(self.method_combo)
        self.run_button = self.make_button('运行', self.run_selected, 'primary', tip='运行当前算法', icon='play')
        self.run_all_button = self.make_button('全部运行', self.run_all, tip='运行全部算法', icon='layers')
        controls.addWidget(self.run_button)
        controls.addWidget(self.run_all_button)
        self.processing_label = self.label('', 'caption')
        controls.addWidget(self.processing_label)
        self.processing_label.hide()
        self.processing_bar = QProgressBar()
        self.processing_bar.setRange(0, 100)
        self.processing_bar.setTextVisible(False)
        self.processing_bar.setFixedHeight(5)
        self.processing_bar.hide()
        controls.addWidget(self.processing_bar)
        self.cancel_button = self.make_button('', self.cancel_filters, icon='stop',
                                             tip='当前算法结束后停止，保留已经完成的结果')
        self.cancel_button.hide()
        controls.addWidget(self.cancel_button)
        sidebar_layout.addWidget(method_card)

        metrics_card, controls = self.card('效果分析', '03')
        self.input_snr_label = self.label('输入 SNR：—')
        self.output_snr_label = self.label('输出 SNR：—')
        self.gain_label = self.label('SNR 变化：—', 'caption')
        self.time_label = self.label('处理耗时：—', 'caption')
        for item in [self.input_snr_label, self.output_snr_label, self.gain_label, self.time_label]:
            controls.addWidget(item)
        self.gain_label.setToolTip('以原始音频为参考；负值表示误差增大')
        self.export_button = self.make_button('', self.export_results, icon='export',
                                              tip='导出 WAV 音频、CSV 指标及图表')
        metrics_card.layout().itemAt(0).layout().addWidget(self.export_button)
        sidebar_layout.addWidget(metrics_card)
        sidebar_layout.addStretch()
        sidebar_scroll = self.scroll_area(sidebar)
        sidebar_scroll.setFixedWidth(292)
        body.addWidget(sidebar_scroll)

        chart_card, chart_layout = self.card(None)
        chart_layout.setContentsMargins(16, 12, 16, 12)
        legend_row = QHBoxLayout()
        for text, color in [('原始', '#718096'), ('加噪', '#36a4c5'), ('降噪', '#3478d4')]:
            label = self.label('●  ' + text, 'caption')
            label.setStyleSheet(f'color: {color};')
            legend_row.addWidget(label)
        legend_row.addStretch()
        reset_button = self.make_button('', self.reset_plot_views, icon='maximize',
                                       tip='恢复全部数据范围；图内可拖动和滚轮缩放')
        legend_row.addWidget(reset_button)
        chart_layout.addLayout(legend_row)
        plots_content = QWidget()
        plots_content.setObjectName('plotsContent')
        plots_layout = QVBoxLayout(plots_content)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(10)
        self.original_plot = self.make_plot('原始波形', '时间', 's')
        self.noisy_plot = self.make_plot('加噪波形', '时间', 's')
        self.result_plot = self.make_plot('降噪波形', '时间', 's')
        self.fft_plot = self.make_plot('频谱对比', '频率', 'Hz')
        self.fft_plot.setLabel('left', '幅度', units='dB')
        self.fft_plot.setToolTip('频谱使用开头最多 32768 个采样点；拖动或滚轮可查看细节')
        for plot in (self.original_plot, self.noisy_plot, self.result_plot, self.fft_plot):
            plots_layout.addWidget(plot, 1)
        chart_layout.addWidget(self.scroll_area(plots_content), 1)
        body.addWidget(chart_card, 1)
        layout.addLayout(body, 1)

        player_card, player_layout = self.card(None)
        player_layout.setContentsMargins(16, 12, 16, 12)
        player_row = QHBoxLayout()
        player_row.setSpacing(8)
        headphone = QLabel()
        headphone.setPixmap(self.icon('headphones').pixmap(20, 20))
        headphone.setToolTip('试听音频')
        player_row.addWidget(headphone)
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(140)
        self.source_combo.currentTextChanged.connect(self.change_source)
        player_row.addWidget(self.source_combo)
        self.back_button = self.make_button('', self.back_five, tip='后退 5 秒', icon='rewind')
        self.play_button = self.make_button('', self.play_audio, 'soft', tip='播放', icon='play')
        self.pause_button = self.make_button('', self.pause_audio, tip='暂停', icon='pause')
        self.stop_button = self.make_button('', self.stop_audio, tip='停止', icon='stop')
        self.forward_button = self.make_button('', self.forward_five, tip='前进 5 秒', icon='forward')
        for button in [self.back_button, self.play_button, self.pause_button, self.stop_button, self.forward_button]:
            player_row.addWidget(button)
        player_row.addStretch()
        self.position_label = self.label('00:00 / 00:00', 'caption')
        player_row.addWidget(self.position_label)
        player_row.addWidget(self.label('×', 'caption'))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(['0.5', '0.75', '1.0', '1.25', '1.5', '2.0'])
        self.speed_combo.setCurrentText('1.0')
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        player_row.addWidget(self.speed_combo)
        self.save_button = self.make_button('', self.save_audio, tip='保存当前音频', icon='save')
        player_row.addWidget(self.save_button)
        player_layout.addLayout(player_row)
        self.progress = QSlider(Qt.Orientation.Horizontal)
        self.progress.setRange(0, 1000)
        self.progress.setAccessibleName('播放进度')
        self.progress.sliderReleased.connect(self.seek_slider)
        player_layout.addWidget(self.progress)
        layout.addWidget(player_card)
        return page

    def make_scene_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(14)
        heading = QHBoxLayout()
        heading.addWidget(self.label('场景声音增强', 'heading'), 1)
        self.scene_import_button = self.make_button('', self.import_audio, tip='导入音频', icon='upload')
        heading.addWidget(self.scene_import_button)
        layout.addLayout(heading)

        body = QHBoxLayout()
        body.setSpacing(16)
        frame, controls = self.card('场景控制')
        frame.setFixedWidth(300)
        controls.setSpacing(14)
        controls.addWidget(self.label('输入音频', 'caption'))
        self.scene_input_combo = QComboBox()
        self.scene_input_combo.setMinimumHeight(38)
        self.scene_input_combo.currentTextChanged.connect(self.scene_parameters_changed)
        controls.addWidget(self.scene_input_combo)
        controls.addWidget(self.label('增强模式', 'caption'))
        grid = QGridLayout()
        grid.setSpacing(10)
        self.scene_group = QButtonGroup(self)
        self.scene_modes = ['突出人声', '突出音乐', '突出环境', '原始平衡']
        self.scene_mode_buttons = []
        for index, (name, icon) in enumerate(zip(self.scene_modes, ['mic', 'music', 'trees', 'sliders'])):
            button = QToolButton()
            button.setText(name)
            button.setIcon(self.icon(icon))
            button.setIconSize(QSize(24, 24))
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            button.setObjectName('sceneMode')
            button.setCheckable(True)
            button.setMinimumHeight(86)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAccessibleName(name)
            self.scene_group.addButton(button, index)
            self.scene_mode_buttons.append(button)
            grid.addWidget(button, index // 2, index % 2)
        self.scene_mode_buttons[0].setChecked(True)
        self.scene_group.idClicked.connect(self.scene_parameters_changed)
        controls.addLayout(grid)
        strength_row = QHBoxLayout()
        strength_row.addWidget(self.label('增强强度', 'caption'), 1)
        self.scene_strength_label = self.label('70%', 'caption')
        strength_row.addWidget(self.scene_strength_label)
        controls.addLayout(strength_row)
        self.scene_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.scene_strength_slider.setRange(0, 100)
        self.scene_strength_slider.setValue(70)
        self.scene_strength_slider.setAccessibleName('增强强度')
        self.scene_strength_slider.setToolTip('0% 保留输入音频，100% 使用完整增强结果')
        self.scene_strength_slider.valueChanged.connect(self.update_scene_strength)
        controls.addWidget(self.scene_strength_slider)
        self.scene_button = self.make_button('运行增强', self.apply_scene, 'primary', icon='play')
        controls.addWidget(self.scene_button)
        self.scene_status = self.label('', 'caption')
        controls.addWidget(self.scene_status)
        self.scene_processing_bar = QProgressBar()
        self.scene_processing_bar.setRange(0, 0)
        self.scene_processing_bar.setTextVisible(False)
        self.scene_processing_bar.setFixedHeight(5)
        self.scene_processing_bar.hide()
        controls.addWidget(self.scene_processing_bar)
        controls.addStretch()
        body.addWidget(self.scroll_area(frame))
        body.itemAt(0).widget().setFixedWidth(308)

        charts = QWidget()
        charts.setObjectName('sidebar')
        charts_layout = QVBoxLayout(charts)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(12)
        self.scene_before_plot = self.make_plot('处理前波形', '时间', 's')
        self.scene_after_plot = self.make_plot('处理后波形', '时间', 's')
        self.scene_fft_plot = self.make_plot('频谱对比', '频率', 'Hz')
        self.scene_fft_plot.setLabel('left', '幅度', units='dB')
        self.scene_fft_plot.setToolTip('频谱使用开头最多 32768 个采样点')
        for plot in [self.scene_before_plot, self.scene_after_plot, self.scene_fft_plot]:
            card, content = self.card(None)
            content.setContentsMargins(10, 8, 10, 8)
            plot.setMinimumHeight(164)
            content.addWidget(plot)
            charts_layout.addWidget(card, 1)
        body.addWidget(self.scroll_area(charts), 1)
        layout.addLayout(body, 1)

        player_card, player_layout = self.card(None)
        player_layout.setContentsMargins(16, 12, 16, 12)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.label('试听与保存', 'section'))
        self.scene_play_button = self.make_button('', self.play_audio, 'soft', tip='播放', icon='play')
        self.scene_pause_button = self.make_button('', self.pause_audio, tip='暂停', icon='pause')
        self.scene_stop_button = self.make_button('', self.stop_audio, tip='停止', icon='stop')
        for button in [self.scene_play_button, self.scene_pause_button, self.scene_stop_button]:
            row.addWidget(button)
        self.scene_audition_combo = QComboBox()
        self.scene_audition_combo.addItems(['处理前', '处理后'])
        self.scene_audition_combo.currentIndexChanged.connect(self.scene_audition_changed)
        row.addWidget(self.scene_audition_combo)
        row.addStretch()
        before = self.label('●  处理前', 'caption')
        before.setStyleSheet('color: #36a4c5;')
        after = self.label('●  处理后', 'caption')
        after.setStyleSheet('color: #3478d4;')
        row.addWidget(before)
        row.addWidget(after)
        reset = self.make_button('', self.reset_scene_views, tip='重置视图', icon='maximize')
        row.addWidget(reset)
        self.scene_position_label = self.label('00:00 / 00:00', 'caption')
        row.addWidget(self.scene_position_label)
        self.scene_save_button = self.make_button('', self.save_scene_audio, tip='保存当前试听音频', icon='export')
        row.addWidget(self.scene_save_button)
        player_layout.addLayout(row)
        self.scene_progress = QSlider(Qt.Orientation.Horizontal)
        self.scene_progress.setRange(0, 1000)
        self.scene_progress.setAccessibleName('场景音频播放进度')
        self.scene_progress.sliderReleased.connect(self.seek_scene_slider)
        player_layout.addWidget(self.scene_progress)
        layout.addWidget(player_card)
        return page

    def make_translation_page(self):
        page = QWidget()
        page.setObjectName('translationPage')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(14)
        heading = QHBoxLayout()
        heading.addWidget(self.label('AI 实时翻译', 'heading'), 1)
        self.translation_state = self.label('就绪', 'badge')
        heading.addWidget(self.translation_state)
        layout.addLayout(heading)

        settings, content = self.card(None)
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)
        grid.addWidget(self.label('输入语言', 'caption'), 0, 0)
        grid.addWidget(self.label('目标语言', 'caption'), 0, 1)
        grid.addWidget(self.label('模式', 'caption'), 0, 2)
        self.input_language_combo = QComboBox()
        self.target_language_combo = QComboBox()
        languages = list(LANGUAGE_CODES)
        self.input_language_combo.addItems(languages)
        self.target_language_combo.addItem('英语')
        self.target_language_combo.setEnabled(False)
        self.target_language_combo.setToolTip('Whisper 的语音翻译目前输出英语')
        grid.addWidget(self.input_language_combo, 1, 0)
        grid.addWidget(self.target_language_combo, 1, 1)
        self.translation_model_combo = QComboBox()
        self.translation_model_combo.addItem('高精度', 'medium')
        self.translation_model_combo.addItem('快速', 'base')
        self.translation_model_combo.setToolTip('高精度使用 medium；快速使用 base')
        grid.addWidget(self.translation_model_combo, 1, 2)
        self.translation_denoise_check = QCheckBox('AI 降噪')
        self.translation_tts_check = QCheckBox('语音播报')
        for control in [self.translation_denoise_check, self.translation_tts_check]:
            control.setEnabled(False)
            control.setToolTip('此功能尚未接入')
        grid.addWidget(self.translation_denoise_check, 2, 0)
        grid.addWidget(self.translation_tts_check, 2, 1)
        content.addLayout(grid)
        layout.addWidget(settings)

        transcripts = QHBoxLayout()
        transcripts.setSpacing(16)
        for name, icon, placeholder in [('原语识别', 'mic', '待识别'), ('翻译结果', 'languages', '待翻译')]:
            card, card_layout = self.card(None)
            row = QHBoxLayout()
            symbol = QLabel()
            symbol.setPixmap(self.icon(icon).pixmap(20, 20))
            row.addWidget(symbol)
            row.addWidget(self.label(name, 'section'), 1)
            card_layout.addLayout(row)
            transcript = QPlainTextEdit()
            transcript.setReadOnly(True)
            transcript.setMaximumBlockCount(500)
            transcript.setPlaceholderText(placeholder)
            transcript.setMinimumHeight(140)
            transcript.setAccessibleName(name)
            if icon == 'mic':
                self.asr_text = transcript
            else:
                self.translation_text = transcript
            card_layout.addWidget(transcript, 1)
            transcripts.addWidget(card, 1)
        layout.addLayout(transcripts, 1)

        waveform_card, content = self.card(None)
        content.setContentsMargins(10, 8, 10, 8)
        self.translation_plot = self.make_plot('音频波形', '时间', 's')
        self.translation_plot.setMinimumHeight(180)
        content.addWidget(self.translation_plot)
        layout.addWidget(waveform_card, 1)

        control_card, controls = self.card(None)
        row = QHBoxLayout()
        self.translation_start_button = self.make_button('开始', self.toggle_translation, 'primary',
                                                       tip='开始 / 停止麦克风翻译', icon='mic')
        self.translation_import_button = self.make_button('本地音频', self.translate_file,
                                                        tip='识别并翻译本地音频', icon='upload')
        row.addWidget(self.translation_start_button, 1)
        row.addWidget(self.translation_import_button, 1)
        controls.addLayout(row)
        layout.addWidget(control_card)
        for label in page.findChildren(QLabel):
            label.setWordWrap(False)
        return self.scroll_area(page)

    def make_plot(self, title, bottom_label, units):
        plot = AudioPlot(title)
        plot.setBackground('#ffffff')
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel('bottom', bottom_label, units=units)
        plot.setLabel('left', '幅度')
        plot.setMinimumHeight(132)
        plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        for orientation in ['left', 'bottom']:
            axis = plot.getAxis(orientation)
            axis.setPen(pg.mkPen('#cbd5e1'))
            axis.setTextPen(pg.mkPen('#64748b'))
            axis.setTickFont(QFont('Noto Sans CJK SC', 9))
            axis.enableAutoSIPrefix(False)
        plot.getAxis('left').setWidth(58)
        plot.getPlotItem().setContentsMargins(2, 2, 6, 2)
        return plot

    def apply_style(self):
        self.setStyleSheet('''            QWidget { color: #334155; font-family: "Noto Sans CJK SC"; font-size: 13px; }
            QWidget#appRoot, QStackedWidget, QScrollArea, QWidget#sidebar, QWidget#translationPage { background: #f3f6fa; }
            QWidget#plotsContent { background: #ffffff; }
            QFrame#header { background: #ffffff; border-bottom: 1px solid #dce4ee; }
            QFrame#card { background: #ffffff; border: 1px solid #dce4ee; border-radius: 5px; }
            QFrame#infoBar { background: #eaf3fc; border: 1px solid #d6e7f7; border-radius: 4px; }
            QLabel { background: transparent; border: none; }
            QLabel[textRole="brand"] { font-size: 23px; font-weight: 700; color: #2c609d; }
            QLabel[textRole="heading"] { font-size: 21px; font-weight: 600; color: #24364b; }
            QLabel[textRole="section"] { font-size: 14px; font-weight: 600; color: #34465e; }
            QLabel[textRole="muted"] { color: #64748b; }
            QLabel[textRole="caption"] { font-size: 12px; color: #64748b; }
            QLabel[textRole="step"] { color: #4a83bb; font-size: 12px; font-weight: 600; }
            QLabel[textRole="badge"] { background: #edf4fc; color: #3266a3; padding: 7px 12px; border-radius: 4px; font-size: 12px; }
            QLabel#plotHint { color: #8b99ab; font-size: 12px; background: transparent; }
            QPushButton { background: #ffffff; border: 1px solid #ced9e5; border-radius: 4px; padding: 0px 13px; color: #42566e; font-weight: 500; }
            QPushButton:hover { background: #f1f7fd; border-color: #83b0de; color: #285f9f; }
            QPushButton:pressed { background: #dceafb; border-color: #5c95cf; }
            QPushButton:focus { border: 2px solid #6499d1; }
            QPushButton[iconOnly="true"] { padding: 0; }
            QPushButton[role="primary"] { background: #477fb8; border-color: #477fb8; color: #ffffff; }
            QPushButton[role="primary"]:hover { background: #396fa6; border-color: #396fa6; }
            QPushButton[role="primary"]:pressed { background: #2f6093; }
            QPushButton[role="soft"] { background: #eaf3fc; color: #3266a3; border-color: #cbdff3; }
            QPushButton[role="soft"]:hover { background: #deecfa; border-color: #83b0de; }
            QPushButton[role="nav"] { border: 1px solid transparent; background: transparent; color: #63748b; padding: 0px 18px; }
            QPushButton[role="nav"]:hover { background: #f3f7fc; }
            QPushButton[role="nav"]:checked { background: #e8f2fc; color: #2e69a6; border-color: #cddff2; font-weight: 600; }
            QPushButton[recording="true"] { background: #fff1f0; color: #b74444; border-color: #ecc1bd; }
            QPushButton:disabled { background: #f1f4f7; border-color: #e2e8f0; color: #94a3b8; }
            QComboBox { background: #ffffff; border: 1px solid #ced9e5; border-radius: 4px; padding: 7px 10px; min-height: 20px; }
            QComboBox:hover, QComboBox:focus { border-color: #6499d1; }
            QComboBox:disabled { color: #94a3b8; background: #f1f4f7; border-color: #e2e8f0; }
            QComboBox::drop-down { border: none; width: 22px; }
            QComboBox::down-arrow { image: url("CHEVRON_PATH"); width: 10px; height: 6px; }
            QComboBox QAbstractItemView { background: #ffffff; color: #334155; selection-background-color: #e8f2fc; selection-color: #2e69a6; border: 1px solid #ced9e5; padding: 4px; }
            QSlider::groove:horizontal { height: 4px; background: #e0e7ef; border-radius: 2px; }
            QSlider::sub-page:horizontal { background: #7daedb; border-radius: 2px; }
            QSlider::handle:horizontal { background: #ffffff; border: 2px solid #5e94c6; width: 12px; margin: -6px 0; border-radius: 3px; }
            QSlider::handle:horizontal:hover { background: #eaf3fc; border-color: #3266a3; }
            QSlider::handle:horizontal:disabled { border-color: #bbc7d5; background: #f1f4f7; }
            QSlider::sub-page:horizontal:disabled { background: #cbd5e1; }
            QProgressBar { background: #e7eef6; border: none; border-radius: 2px; }
            QProgressBar::chunk { background: #7daedb; border-radius: 2px; }
            QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
            QScrollBar::handle:vertical { background: #cbd6e3; min-height: 30px; border-radius: 4px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
            QStatusBar { background: #ffffff; border-top: 1px solid #dce4ee; color: #64748b; font-size: 12px; }
            QStatusBar::item { border: none; }
            QToolTip { background: #ffffff; color: #334155; border: 1px solid #ced9e5; padding: 6px; }
            QToolButton#sceneMode { background: #ffffff; border: 1px solid #dce4ee; border-radius: 4px; padding: 10px; color: #576b82; }
            QToolButton#sceneMode:hover { background: #f2f7fd; border-color: #83b0de; }
            QToolButton#sceneMode:checked { background: #eaf3fc; border: 2px solid #83b0de; color: #3266a3; }
            QToolButton#sceneMode:focus { border: 2px solid #477fb8; }
            QToolButton#sceneMode:disabled { background: #f1f4f7; color: #94a3b8; }
            QPlainTextEdit { background: #ffffff; color: #334155; border: none; font-size: 15px; padding: 4px; }
            QCheckBox { spacing: 8px; color: #64748b; }
            QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #cbd5e1; border-radius: 3px; background: #f1f4f7; }
            QCheckBox:disabled { color: #94a3b8; }
            QMessageBox, QFileDialog { background: #ffffff; }
        '''.replace('CHEVRON_PATH', str(Path(__file__).parent / 'icons' / 'chevron-down.svg')))

    def set_feedback(self, message, badge=None):
        self.statusBar().showMessage(message)
        if badge:
            self.state_badge.setText('●  ' + badge)

    def current_scene_mode(self):
        return self.scene_modes[self.scene_group.checkedId()]

    def scene_result_valid(self):
        return (self.scene_audio is not None
                and self.get_source(self.scene_input_combo.currentText()) is self.scene_input_reference
                and self.current_scene_mode() == self.scene_result_mode
                and self.scene_strength_slider.value() == self.scene_result_strength)

    def get_scene_audition(self):
        if self.scene_audition_combo.currentIndex() == 1:
            return self.scene_audio if self.scene_result_valid() else None
        return self.get_source(self.scene_input_combo.currentText())

    def sync_scene_inputs(self):
        selected = self.scene_input_combo.currentText()
        names = [name for name in self.available_sources() if name != '场景增强']
        self.scene_input_combo.blockSignals(True)
        self.scene_input_combo.clear()
        self.scene_input_combo.addItems(names)
        if selected in names:
            self.scene_input_combo.setCurrentText(selected)
        self.scene_input_combo.blockSignals(False)
        self.scene_parameters_changed()

    def scene_parameters_changed(self, *unused):
        if self.current_scene_mode() == '原始平衡' and self.original is not None:
            self.scene_input_combo.blockSignals(True)
            self.scene_input_combo.setCurrentText('原始音频')
            self.scene_input_combo.blockSignals(False)
        self.scene_strength_label.setText(f'{self.scene_strength_slider.value()}%')
        valid = self.scene_result_valid()
        self.scene_audition_combo.model().item(1).setEnabled(valid)
        if not valid and self.scene_audition_combo.currentIndex() == 1:
            self.scene_audition_combo.blockSignals(True)
            self.scene_audition_combo.setCurrentIndex(0)
            self.scene_audition_combo.blockSignals(False)
        if self.worker is None:
            self.scene_status.setText('完成' if valid else '待运行' if self.scene_audio is not None else '')
        self.refresh_scene_graphs()
        if self.pages.currentIndex() == 1:
            self.load_active_audio()
        else:
            self.update_controls()

    def update_scene_strength(self, value=None):
        self.scene_parameters_changed()

    def scene_audition_changed(self, *unused):
        if self.pages.currentIndex() == 1:
            self.load_active_audio()

    def load_active_audio(self):
        index = self.pages.currentIndex()
        audio = (self.get_scene_audition() if index == 1 else
                 self.get_source(self.source_combo.currentText()) if index == 0 else None)
        self.previous_playing = False
        was_playing = self.player.playing
        self.player.load(audio)
        if was_playing:
            self.set_feedback('已停止播放', '就绪')
        self.update_progress()

    def refresh_scene_graphs(self):
        before = self.get_source(self.scene_input_combo.currentText())
        after = self.scene_audio if self.scene_result_valid() else None
        peak = max((float(np.max(np.abs(audio))) for audio in [before, after]
                    if audio is not None and len(audio)), default=1.0)
        self.scene_waveform_limit = max(peak * 1.1, 1e-5)
        for plot, audio, color in [(self.scene_before_plot, before, '#36a4c5'),
                                   (self.scene_after_plot, after, '#3478d4')]:
            plot.clear()
            plot.empty_hint.setVisible(audio is None)
            plot.empty_hint.setText('未加载音频' if plot is self.scene_before_plot else '待增强')
            if audio is not None:
                x, y = waveform(audio, self.fs)
                plot.plot(x, y, pen=pg.mkPen(color, width=1.2))
            plot.enableAutoRange(axis='x')
            plot.setYRange(-self.scene_waveform_limit, self.scene_waveform_limit, padding=0)
        self.scene_fft_plot.clear()
        self.scene_fft_plot.empty_hint.setVisible(before is None)
        for audio, color in [(before, '#36a4c5'), (after, '#3478d4')]:
            if audio is not None:
                f, db = fft_spectrum(audio, self.fs)
                self.scene_fft_plot.plot(f, db, pen=pg.mkPen(color, width=1.2))
        self.scene_fft_plot.enableAutoRange(axis='y')
        self.scene_fft_plot.setXRange(0, min(8000, self.fs / 2), padding=0)

    def reset_scene_views(self):
        for plot in [self.scene_before_plot, self.scene_after_plot, self.scene_fft_plot]:
            plot.enableAutoRange()
            plot.autoRange()
        for plot in [self.scene_before_plot, self.scene_after_plot]:
            plot.setYRange(-self.scene_waveform_limit, self.scene_waveform_limit, padding=0)
        self.scene_fft_plot.setXRange(0, min(8000, self.fs / 2), padding=0)
        self.set_feedback('视图已重置')

    def refresh_translation_graph(self):
        self.translation_plot.clear()
        audio = self.translation_audio if self.translation_audio is not None else self.original
        fs = self.translation_audio_fs if self.translation_audio is not None else self.fs
        self.translation_plot.empty_hint.setVisible(audio is None)
        if audio is not None:
            x, y = waveform(audio, fs)
            self.translation_plot.plot(x, y, pen=pg.mkPen('#3478d4', width=1.2))
        self.translation_plot.enableAutoRange()

    def toggle_translation(self):
        if self.translation_worker is not None:
            self.stop_translation()
        else:
            self.start_translation()

    def translate_file(self):
        if self.is_busy():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, '选择音频', '', '音频 (*.wav *.mp3 *.flac *.ogg *.aiff *.m4a)'
        )
        if path:
            self.start_translation(path)

    def start_translation(self, audio_path=None):
        if self.is_busy():
            return
        self.player.stop()
        self.previous_playing = False
        language = LANGUAGE_CODES[self.input_language_combo.currentText()]
        self.translation_worker = TranslationWorker(
            language=language, audio_path=audio_path,
            model_size=self.translation_model_combo.currentData()
        )
        self.translation_worker.result_signal.connect(self.translation_result_received)
        self.translation_worker.status_signal.connect(self.translation_status_received)
        self.translation_worker.error_signal.connect(self.translation_error_received)
        self.translation_worker.audio_signal.connect(self.translation_audio_received)
        self.translation_worker.finished.connect(self.translation_finished)
        self.translation_stopping = False
        self.translation_error = ''
        self.translation_audio = None
        self.asr_text.clear()
        self.translation_text.clear()
        self.translation_plot.clear()
        self.translation_plot.empty_hint.show()
        self.translation_plot.empty_hint.setText('等待音频')
        self.translation_start_button.setText('停止')
        self.translation_start_button.setIcon(self.icon('stop', white=True))
        self.translation_status_received('准备中')
        self.update_controls()
        self.translation_worker.start()

    def stop_translation(self, finish_pending=True):
        if self.translation_worker is None:
            return
        self.translation_stopping = True
        self.translation_worker.stop(finish_pending=finish_pending)
        self.translation_status_received('停止中')
        self.update_controls()

    def translation_status_received(self, status):
        if self.translation_stopping:
            status = '停止中'
        self.translation_state.setText(status)
        if not self.asr_text.toPlainText():
            self.asr_text.setPlaceholderText(status if status in ['未收到声音', '未检测到语音'] else '待识别')
        self.set_feedback(status, status)

    def translation_result_received(self, original, translated):
        if original:
            self.asr_text.appendPlainText(original)
        if translated:
            self.translation_text.appendPlainText(translated)

    def translation_audio_received(self, audio, fs):
        if self.translation_worker is None or self.translation_stopping:
            return
        if self.translation_worker.audio_path is None:
            previous = self.translation_audio if self.translation_audio is not None else np.empty(0, dtype='float32')
            self.translation_audio = np.concatenate((previous, audio))[-fs * 3:]
        else:
            self.translation_audio = audio
        self.translation_audio_fs = fs
        self.refresh_translation_graph()

    def translation_error_received(self, message):
        self.translation_error = message
        self.translation_state.setText('失败')
        self.set_feedback('翻译失败', '翻译失败')
        if not self.close_after_translation:
            QMessageBox.critical(self, '翻译失败', message)

    def translation_finished(self):
        worker = self.translation_worker
        if worker is None:
            return
        state = ('失败' if self.translation_error else '已停止' if self.translation_stopping
                 else '完成' if worker.audio_path is not None else '已停止')
        has_result = bool(self.asr_text.toPlainText() or self.translation_text.toPlainText())
        if state == '完成' and not has_result:
            state = '未检测到语音'
        self.translation_worker = None
        self.translation_stopping = False
        self.translation_start_button.setText('开始')
        self.translation_start_button.setIcon(self.icon('mic', white=True))
        self.translation_state.setText(state)
        self.set_feedback(state if has_result or self.translation_error else state + ' · 未识别到语音', state)
        self.update_controls()
        worker.deleteLater()
        if self.close_after_translation:
            QTimer.singleShot(0, self.close)

    def update_controls(self):
        translating = self.translation_worker is not None
        busy = self.worker is not None or translating
        recording = self.recorder.stream is not None
        has_audio = self.original is not None
        editable = has_audio and not busy and not recording
        self.record_button.setEnabled(not busy)
        self.import_button.setEnabled(not busy and not recording)
        for control in [self.noise_button, self.noise_combo, self.noise_slider,
                        self.method_combo, self.run_button, self.run_all_button,
                        self.scene_button, self.scene_input_combo, self.scene_strength_slider,
                        *self.scene_mode_buttons]:
            control.setEnabled(editable)
        self.scene_input_combo.setEnabled(editable and self.current_scene_mode() != '原始平衡')
        self.export_button.setEnabled(editable)
        playable = self.player.audio is not None and not recording and not translating
        for control in [self.source_combo, self.back_button, self.forward_button,
                        self.speed_combo, self.progress, self.save_button]:
            control.setEnabled(playable)
        self.play_button.setEnabled(playable and not self.player.playing)
        self.pause_button.setEnabled(playable and self.player.playing)
        self.stop_button.setEnabled(playable and (self.player.playing or self.player.position > 0))
        self.scene_import_button.setEnabled(not busy and not recording)
        self.translation_import_button.setEnabled(not busy and not recording)
        self.translation_start_button.setEnabled(
            not self.translation_stopping and (translating or (self.worker is None and not recording))
        )
        self.input_language_combo.setEnabled(not translating)
        self.translation_model_combo.setEnabled(not translating)
        scene_playable = self.get_scene_audition() is not None and not recording and not translating
        for control in [self.scene_audition_combo, self.scene_progress, self.scene_save_button]:
            control.setEnabled(scene_playable)
        active_scene = self.pages.currentIndex() == 1
        self.scene_play_button.setEnabled(scene_playable and not (active_scene and self.player.playing))
        self.scene_pause_button.setEnabled(scene_playable and active_scene and self.player.playing)
        self.scene_stop_button.setEnabled(scene_playable and active_scene and (self.player.playing or self.player.position > 0))

    def confirm_replace(self, action):
        if not self.results and self.scene_audio is None:
            return True
        return QMessageBox.question(
            self, '替换已有结果',
            action + '会清除当前降噪和场景处理结果。\n需要保留时，请先导出结果。是否继续？',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.Yes

    def reset_plot_views(self):
        for plot in [self.original_plot, self.noisy_plot, self.result_plot, self.fft_plot]:
            plot.enableAutoRange()
            plot.autoRange()
        for plot in [self.original_plot, self.noisy_plot, self.result_plot]:
            plot.setYRange(-self.waveform_limit, self.waveform_limit, padding=0)
        self.fft_plot.setXRange(0, min(8000, self.fs / 2), padding=0)
        self.set_feedback('视图已重置')

    def is_busy(self):
        if self.translation_worker is not None:
            self.set_feedback('正在翻译，请先停止翻译')
            return True
        if self.worker is not None:
            self.set_feedback('处理中')
            return True
        if self.recorder.stream is not None:
            self.set_feedback('录音中')
            return True
        return False

    def toggle_record(self):
        if self.recorder.stream is None:
            if self.is_busy():
                return
            if not self.confirm_replace('重新录音'):
                return
            try:
                self.player.stop()
                self.recorder.start()
                self.record_started = time.monotonic()
                self.record_button.setText('结束')
                self.record_button.setIcon(self.icon('stop'))
                self.record_button.setProperty('recording', True)
                self.record_button.style().unpolish(self.record_button)
                self.record_button.style().polish(self.record_button)
                self.set_feedback('录音中', '录音中')
            except Exception as error:
                self.set_feedback('录音失败', '录音失败')
                QMessageBox.critical(self, '录音失败', str(error))
        else:
            try:
                audio = self.recorder.stop()
                self.set_audio(audio, '麦克风录音')
            except Exception as error:
                self.set_feedback('录音未保存', '录音失败')
                QMessageBox.critical(self, '停止录音失败', str(error))
            finally:
                recording = self.recorder.stream is not None
                self.record_button.setText('结束' if recording else '录音')
                self.record_button.setIcon(self.icon('stop') if recording else self.icon('mic', white=True))
                self.record_button.setProperty('recording', recording)
                self.record_button.style().unpolish(self.record_button)
                self.record_button.style().polish(self.record_button)
        self.update_controls()

    def import_audio(self):
        if self.is_busy() or self.recorder.stream is not None:
            return
        path, unused_filter = QFileDialog.getOpenFileName(
            self, '选择音频', '', '音频 (*.wav *.flac *.ogg *.aiff)'
        )
        if not path:
            return
        try:
            self.set_feedback('正在读取音频…', '导入中')
            samples = read_audio(path, self.fs)
            if not self.confirm_replace('导入新音频'):
                self.set_feedback('已取消导入', '就绪')
                return
            self.set_audio(samples, Path(path).name)
        except Exception as error:
            self.set_feedback('导入失败', '导入失败')
            QMessageBox.critical(self, '导入失败', str(error))

    def set_audio(self, audio, name):
        if len(audio) < 128:
            raise ValueError('录音时间过短')
        if np.asarray(audio).ndim != 1 or not np.all(np.isfinite(audio)):
            raise ValueError('音频必须是一维数据，且不能包含无效数值')
        self.player.stop()
        self.original = audio.copy()
        self.translation_audio = None
        self.noisy = None
        self.added_noise = None
        self.results.clear()
        self.times.clear()
        self.scene_audio = None
        self.audio_label.setText(name)
        self.audio_label.setToolTip(name)
        self.audio_meta.setText(f'{len(audio) / self.fs:.2f} s · {self.fs / 1000:g} kHz')
        self.processing_label.clear()
        self.processing_label.hide()
        self.scene_status.clear()
        self.update_sources('原始音频')
        self.set_feedback('音频已加载', '就绪')

    def update_noise_text(self):
        value = self.noise_slider.value()
        if self.noise_combo.currentText() == '高斯噪声':
            self.noise_value_label.setText(f'目标 SNR：{value} dB')
            self.noise_slider.setToolTip('SNR 越低，噪声越强')
        else:
            self.noise_value_label.setText(f'脉冲概率：{value / 10:.1f}%')
            self.noise_slider.setToolTip('概率越高，脉冲越频繁')

    def add_noise(self):
        if self.is_busy():
            return
        if self.original is None:
            QMessageBox.information(self, '提示', '请先录音或导入音频')
            return
        if not self.confirm_replace('重新添加噪声'):
            return
        try:
            value = self.noise_slider.value()
            if self.noise_combo.currentText() == '高斯噪声':
                self.noisy, self.added_noise = gaussian_noise(self.original, value)
            else:
                self.noisy, self.added_noise = impulse_noise(self.original, value / 10)
            self.results.clear()
            self.times.clear()
            self.scene_audio = None
            self.processing_label.clear()
            self.processing_label.hide()
            self.scene_status.clear()
            self.update_sources('加噪音频')
            self.set_feedback('噪声已添加', '就绪')
        except Exception as error:
            self.set_feedback('加噪失败', '操作失败')
            QMessageBox.critical(self, '加噪失败', str(error))

    def run_selected(self):
        self.start_filters([self.method_combo.currentText()])

    def run_all(self):
        self.start_filters(METHODS)

    def start_filters(self, methods):
        if self.is_busy():
            return
        audio = self.noisy if self.noisy is not None else self.original
        if audio is None:
            QMessageBox.information(self, '提示', '请先录音或导入音频')
            return
        self.worker = FilterThread(methods, audio, self.fs, self.added_noise)
        self.begin_processing()

    def begin_processing(self):
        self.worker.finished.connect(self.filter_finished)
        self.worker.progress.connect(self.processing_progress)
        self.processing_label.setText('准备中…')
        self.processing_label.show()
        self.processing_bar.setRange(0, 0)
        self.processing_bar.show()
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(len(self.worker.methods) > 1)
        self.run_button.setText('处理中…')
        if self.worker.scene:
            self.scene_button.setText('处理中…')
            self.scene_status.setText('处理中…')
            self.scene_processing_bar.show()
        self.update_controls()
        self.set_feedback('处理中', '处理中')
        self.worker.start()

    def processing_progress(self, name, completed, total):
        if self.worker is None or self.worker.isInterruptionRequested():
            return
        text = f'{name} · {completed}/{total}'
        self.processing_label.setText(text)
        if self.worker.scene:
            self.scene_status.setText(text)
        self.set_feedback('正在处理 · ' + text, '处理中')

    def cancel_filters(self):
        if self.worker is None:
            return
        self.worker.requestInterruption()
        self.cancel_button.setEnabled(False)
        self.processing_label.setText('当前算法结束后停止')
        self.set_feedback('等待当前算法结束', '等待停止')

    def filter_finished(self):
        worker = self.worker
        if worker is None:
            return
        if worker.results:
            last_name = list(worker.results)[-1]
            if worker.scene:
                self.scene_audio = worker.results[last_name]
                self.scene_input_reference = worker.input_reference
                self.scene_result_mode = last_name
                self.scene_result_strength = round(worker.strength * 100)
                self.times['场景增强'] = worker.times[last_name]
                self.scene_status.setText(last_name + ' · 完成')
                self.scene_audition_combo.blockSignals(True)
                self.scene_audition_combo.setCurrentIndex(1)
                self.scene_audition_combo.blockSignals(False)
                preferred = '场景增强'
            else:
                self.results.update(worker.results)
                self.times.update(worker.times)
                preferred = last_name
            self.update_sources(preferred)
        self.worker = None
        self.processing_bar.hide()
        self.scene_processing_bar.hide()
        self.cancel_button.hide()
        self.run_button.setText('运行')
        self.scene_button.setText('运行增强')
        self.update_controls()
        if worker.error:
            self.processing_label.setText('处理失败，可重试')
            if worker.scene:
                self.scene_status.setText('处理失败，可重试')
            self.set_feedback('处理失败，已完成结果保留', '处理失败')
            QMessageBox.critical(self, '处理失败', worker.error)
        elif worker.isInterruptionRequested():
            self.processing_label.setText(f'已停止 · {len(worker.results)} 个结果')
            self.set_feedback('已停止', '已停止')
        else:
            self.processing_label.setText(f'完成 · {len(worker.results)} 个结果')
            self.set_feedback('处理完成', '处理完成')
        worker.deleteLater()

    def apply_scene(self):
        if self.is_busy():
            return
        name = self.current_scene_mode()
        input_name = self.scene_input_combo.currentText()
        audio = self.get_source(input_name)
        if audio is None:
            QMessageBox.information(self, '提示', '请先录音或导入音频')
            return
        noise = self.added_noise if input_name == '加噪音频' else None
        self.worker = FilterThread([name], audio, self.fs, noise, scene=True,
                                   strength=self.scene_strength_slider.value() / 100)
        self.worker.input_reference = audio
        self.begin_processing()

    def available_sources(self):
        sources = []
        if self.original is not None:
            sources.append('原始音频')
        if self.noisy is not None:
            sources.append('加噪音频')
        sources.extend(self.results.keys())
        if self.scene_audio is not None:
            sources.append('场景增强')
        return sources

    def get_source(self, name):
        if name == '原始音频':
            return self.original
        if name == '加噪音频':
            return self.noisy
        if name == '场景增强':
            return self.scene_audio
        return self.results.get(name)

    def update_sources(self, preferred):
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItems(self.available_sources())
        self.source_combo.setCurrentText(preferred)
        self.source_combo.blockSignals(False)
        self.sync_scene_inputs()
        self.refresh_translation_graph()
        self.change_source(preferred)

    def change_source(self, name):
        self.refresh_graphs()
        self.load_active_audio()

    def refresh_graphs(self):
        name = self.source_combo.currentText()
        result = self.results.get(name)
        if name == '场景增强':
            result = self.scene_audio
        plots = [
            (self.original_plot, self.original, '#718096'),
            (self.noisy_plot, self.noisy, '#36a4c5'),
            (self.result_plot, result, '#3478d4')
        ]
        peak = max((float(np.max(np.abs(audio))) for _, audio, _ in plots
                    if audio is not None and len(audio)), default=1.0)
        self.waveform_limit = max(peak * 1.1, 1e-5)
        order = 10 ** np.floor(np.log10(self.waveform_limit))
        ratio = self.waveform_limit / order
        tick_step = order * (5 if ratio >= 5 else 2 if ratio >= 2 else 1)
        for plot, audio, color in plots:
            plot.clear()
            plot.empty_hint.setVisible(audio is None)
            if audio is not None:
                x, y = waveform(audio, self.fs)
                plot.plot(x, y, pen=pg.mkPen(color, width=1.1))
            plot.enableAutoRange()
            plot.setYRange(-self.waveform_limit, self.waveform_limit, padding=0)
            plot.getAxis('left').setTicks([[
                (-tick_step, f'{-tick_step:g}'), (0, '0'), (tick_step, f'{tick_step:g}')
            ]])
        self.original_plot.empty_hint.setText('未加载音频')
        self.noisy_plot.empty_hint.setText('未添加噪声')
        self.result_plot.empty_hint.setText('未选择降噪结果')
        self.fft_plot.clear()
        self.fft_plot.empty_hint.setVisible(self.original is None)
        for audio, color in [(self.original, '#718096'), (self.noisy, '#36a4c5'), (result, '#3478d4')]:
            if audio is not None:
                f, db = fft_spectrum(audio, self.fs)
                self.fft_plot.plot(f, db, pen=pg.mkPen(color, width=1.2))
        self.fft_plot.enableAutoRange(axis='y')
        self.fft_plot.setXRange(0, min(8000, self.fs / 2), padding=0)
        input_snr = calculate_snr(self.original, self.noisy) if self.noisy is not None else None
        output_snr = calculate_snr(self.original, result) if input_snr is not None and result is not None else None
        self.input_snr_label.setText('输入 SNR：' + snr_text(input_snr))
        self.output_snr_label.setText('输出 SNR：' + snr_text(output_snr))
        self.gain_label.setText('SNR 变化：—')
        self.gain_label.setStyleSheet('color: #64748b;')
        if output_snr is not None and input_snr is not None:
            change = output_snr - input_snr
            if np.isnan(change):
                self.gain_label.setText('SNR 变化：—')
            else:
                self.gain_label.setText(f'SNR 变化：{change:+.2f} dB' + (' · 误差增大' if change < 0 else ''))
                self.gain_label.setStyleSheet('color: #ad633a;' if change < 0 else 'color: #3266a3;')
        seconds = self.times.get(name)
        self.time_label.setText('处理耗时：—' if seconds is None else f'处理耗时：{seconds:.3f} 秒')

    def play_audio(self):
        if self.recorder.stream is not None:
            return
        try:
            self.player.play()
            name = (self.scene_audition_combo.currentText() if self.pages.currentIndex() == 1
                    else self.source_combo.currentText())
            self.set_feedback('正在播放 · ' + name, '播放中')
            self.update_progress()
        except Exception as error:
            self.set_feedback('播放失败', '播放失败')
            QMessageBox.critical(self, '播放失败', str(error))

    def pause_audio(self):
        self.player.pause()
        self.previous_playing = False
        self.set_feedback('已暂停', '已暂停')
        self.update_progress()

    def stop_audio(self):
        self.player.stop()
        self.previous_playing = False
        self.set_feedback('已停止播放', '就绪')
        self.update_progress()

    def back_five(self):
        self.player.seek(self.player.current_time() - 5)
        self.update_progress()

    def forward_five(self):
        self.player.seek(self.player.current_time() + 5)
        self.update_progress()

    def change_speed(self, text):
        self.player.set_speed(float(text))

    def seek_slider(self):
        seconds = self.player.duration() * self.progress.value() / 1000
        self.player.seek(seconds)

    def seek_scene_slider(self):
        self.player.seek(self.player.duration() * self.scene_progress.value() / 1000)
        self.update_progress()

    def update_progress(self):
        current = self.player.current_time()
        total = self.player.duration()
        if not self.progress.isSliderDown():
            value = 0 if total == 0 else int(1000 * current / total)
            self.progress.setValue(value)
        self.position_label.setText(f'{int(current//60):02d}:{int(current%60):02d} / {int(total//60):02d}:{int(total%60):02d}')
        if not self.scene_progress.isSliderDown():
            self.scene_progress.setValue(0 if total == 0 else int(1000 * current / total))
        self.scene_position_label.setText(self.position_label.text())
        if self.recorder.stream is not None:
            elapsed = time.monotonic() - self.record_started
            self.state_badge.setText(f'●  录音 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}')
        elif self.worker is None:
            if self.player.playing:
                self.state_badge.setText('●  播放中')
            elif self.previous_playing and current >= total:
                self.set_feedback('播放结束', '就绪')
        self.previous_playing = self.player.playing
        self.update_controls()

    def save_audio(self):
        audio = self.get_source(self.source_combo.currentText())
        self.save_audio_file(audio, 'audio.wav')

    def save_scene_audio(self):
        name = 'enhanced.wav' if self.scene_audition_combo.currentIndex() == 1 else 'scene_input.wav'
        self.save_audio_file(self.get_scene_audition(), name)

    def save_audio_file(self, audio, default_name):
        if audio is None:
            return
        path, unused_filter = QFileDialog.getSaveFileName(self, '保存音频', default_name, 'WAV (*.wav)')
        if path:
            try:
                if not Path(path).suffix:
                    path += '.wav'
                write_audio(path, audio, self.fs)
                self.set_feedback('音频已保存 · ' + Path(path).name, '已保存')
            except Exception as error:
                self.set_feedback('保存失败', '保存失败')
                QMessageBox.critical(self, '保存失败', str(error))

    def export_results(self):
        if self.original is None:
            return
        folder = QFileDialog.getExistingDirectory(self, '选择导出文件夹')
        if not folder:
            return
        destination = Path(folder)
        output = {'original': self.original}
        if self.noisy is not None:
            output['noisy'] = self.noisy
        for name, audio in self.results.items():
            output[name] = audio
        if self.scene_audio is not None:
            output['scene'] = self.scene_audio
        filenames = [name + '.wav' for name in output]
        filenames.extend(['metrics.csv', 'original_wave.png', 'noisy_wave.png', 'result_wave.png', 'fft.png'])
        if any((destination / name).exists() for name in filenames):
            answer = QMessageBox.question(
                self, '覆盖已有文件', '此文件夹已有同名导出文件。是否覆盖？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.set_feedback('已取消导出')
                return
        try:
            for name, audio in output.items():
                write_audio(str(destination / (name + '.wav')), audio, self.fs)
            with open(destination / 'metrics.csv', 'w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerow(['算法', '输入SNR(dB)', '输出SNR(dB)', '耗时(s)'])
                input_snr = calculate_snr(self.original, self.noisy) if self.noisy is not None else None
                for name, audio in self.results.items():
                    output_snr = calculate_snr(self.original, audio) if input_snr is not None else None
                    writer.writerow([name, input_snr, output_snr, self.times[name]])
            for name, plot in [('original_wave.png', self.original_plot), ('noisy_wave.png', self.noisy_plot), ('result_wave.png', self.result_plot), ('fft.png', self.fft_plot)]:
                if not plot.grab().save(str(destination / name)):
                    raise OSError('图表保存失败：' + name)
            self.set_feedback('导出完成', '已导出')
            QMessageBox.information(self, '完成', '已保存 WAV、CSV 和图表文件')
        except Exception as error:
            self.set_feedback('导出失败', '导出失败')
            QMessageBox.critical(self, '导出失败', str(error))

    def closeEvent(self, event):
        if self.close_after_translation and self.translation_worker is not None:
            event.ignore()
            return
        if self.recorder.stream is not None or self.worker is not None or self.translation_worker is not None:
            answer = QMessageBox.question(
                self, '结束当前操作',
                '仍在录音、处理或翻译。是否结束当前操作并退出？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self.translation_worker is not None:
            self.close_after_translation = True
            self.stop_translation(finish_pending=False)
            event.ignore()
            return
        self.timer.stop()
        if self.recorder.stream is not None:
            self.recorder.stop()
        self.player.stop()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()
        event.accept()
