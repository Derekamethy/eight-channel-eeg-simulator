"""PySide6 desktop interface for the mock eight-channel EEG simulator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTime, QTimer
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
)

from .dac_model import VirtualDAC
from .device import DeviceState, MockSimulatorDevice
from .eeg_data import EEGData, load_eeg_file
from .fault_conditions import ContactState, ElectrodeConditionModel, InterferenceMode
from .playback import PlaybackClock, PlaybackState
from .synthetic_eeg import ACTIVE_CHANNELS, generate_synthetic_eeg
from .validation import (
    DEFAULT_ACQUISITION_THRESHOLDS,
    DEFAULT_DETECTION_THRESHOLDS,
    DEFAULT_WAVEFORM_THRESHOLDS,
    AcquisitionValidationResult,
    DetectionValidationResult,
    ValidationSuiteResult,
    WaveformValidationResult,
    run_validation_suite,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_eeg_demo.npz"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Eight-Channel EEG Simulator | Engineering Prototype")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 720)

        self.data: EEGData | None = None
        self.loaded_path: Path | None = None
        self.dac = VirtualDAC()
        self.playback = PlaybackClock()
        self.device = MockSimulatorDevice(self._append_log)
        self._prepared_signature: tuple[object, ...] | None = None
        self.plot_curves: dict[str, pg.PlotDataItem] = {}
        self.cursor_lines: list[pg.InfiniteLine] = []
        self.event_regions: list[pg.LinearRegionItem] = []
        self.plot_items: list[pg.PlotItem] = []
        self.conditions = ElectrodeConditionModel(ACTIVE_CHANNELS)
        self.condition_combos: dict[str, QComboBox] = {}
        self._conditioned_cache: np.ndarray | None = None
        self._conditioned_cache_signature: tuple[object, ...] | None = None

        self._build_ui()
        self._apply_style()
        self._update_device_status()
        self._update_controls()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(33)
        self.refresh_timer.timeout.connect(self._refresh_playback)
        self.refresh_timer.start()

    def _build_ui(self) -> None:
        signal_chain_action = QAction("Signal Chain", self)
        signal_chain_action.triggered.connect(self.show_signal_chain)
        self.menuBar().addMenu("Prototype").addAction(signal_chain_action)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(14, 10, 14, 10)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("8-Channel EEG Simulator")
        title.setObjectName("appTitle")
        subtitle = QLabel("Desktop control proof of concept  |  synthetic engineering data")
        subtitle.setObjectName("subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block)
        header.addStretch(1)
        self.mode_badge = QLabel("MOCK  •  NOT CLINICAL")
        self.mode_badge.setObjectName("modeBadge")
        header.addWidget(self.mode_badge, alignment=Qt.AlignmentFlag.AlignTop)
        outer.addLayout(header)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setChildrenCollapsible(False)
        body.addWidget(self._build_control_panel())
        body.addWidget(self._build_monitor_panel())
        body.setSizes([340, 1040])
        outer.addWidget(body, stretch=1)

        footer = QHBoxLayout()
        self.position_label = QLabel("Position  00:00.000 / 00:00.000")
        self.position_label.setObjectName("footerStatus")
        footer.addWidget(self.position_label)
        footer.addStretch(1)
        limitation = QLabel("Future MCU / DAC / analogue stage / acquisition interface are not implemented")
        limitation.setObjectName("footerNote")
        footer.addWidget(limitation)
        outer.addLayout(footer)

        self.setCentralWidget(central)

    def _build_control_panel(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(6)

        device_group = QGroupBox("DEVICE")
        device_layout = QGridLayout(device_group)
        device_layout.setContentsMargins(8, 7, 8, 8)
        device_layout.setHorizontalSpacing(8)
        device_layout.setVerticalSpacing(4)
        device_name = QLabel("Mock EEG Simulator")
        device_name.setObjectName("deviceName")
        device_layout.addWidget(device_name, 0, 0)
        self.device_state_label = QLabel("DISCONNECTED")
        device_layout.addWidget(
            self.device_state_label, 0, 1, alignment=Qt.AlignmentFlag.AlignRight
        )
        self.connection_label = QLabel("● Disconnected")
        self.connection_label.setObjectName("connectionState")
        device_layout.addWidget(self.connection_label, 1, 0)
        device_layout.addWidget(QLabel("Mode:  MOCK"), 1, 1)
        self.connect_button = QPushButton("Connect")
        self.disconnect_button = QPushButton("Disconnect")
        self.connect_button.clicked.connect(self.connect_device)
        self.disconnect_button.clicked.connect(self.disconnect_device)
        device_layout.addWidget(self.connect_button, 2, 0)
        device_layout.addWidget(self.disconnect_button, 2, 1)
        layout.addWidget(device_group)

        file_group = QGroupBox("EEG FILE")
        file_layout = QGridLayout(file_group)
        file_layout.setContentsMargins(8, 7, 8, 8)
        file_layout.setHorizontalSpacing(7)
        file_layout.setVerticalSpacing(4)
        self.generate_button = QPushButton("Generate Synthetic EEG")
        self.generate_button.clicked.connect(self.generate_synthetic)
        self.load_button = QPushButton("Load EEG")
        self.load_button.clicked.connect(self.load_eeg)
        file_layout.addWidget(self.generate_button, 0, 0, 1, 2)
        file_layout.addWidget(self.load_button, 0, 2, 1, 2)
        self.file_name_label = QLabel("No file loaded")
        self.file_name_label.setWordWrap(True)
        self.file_rate_label = QLabel("-")
        self.file_duration_label = QLabel("-")
        self.file_channels_label = QLabel("-")
        self.file_samples_label = QLabel("-")
        file_layout.addWidget(QLabel("File"), 1, 0)
        file_layout.addWidget(self.file_name_label, 1, 1, 1, 3)
        file_layout.addWidget(QLabel("Sample rate"), 2, 0)
        file_layout.addWidget(self.file_rate_label, 2, 1)
        file_layout.addWidget(QLabel("Duration"), 2, 2)
        file_layout.addWidget(self.file_duration_label, 2, 3)
        file_layout.addWidget(QLabel("Channels"), 3, 0)
        file_layout.addWidget(self.file_channels_label, 3, 1)
        file_layout.addWidget(QLabel("Samples/ch"), 3, 2)
        file_layout.addWidget(self.file_samples_label, 3, 3)
        layout.addWidget(file_group)

        channel_group = QGroupBox("ELECTRODES / CHANNELS")
        channel_layout = QGridLayout(channel_group)
        channel_layout.setContentsMargins(8, 7, 8, 8)
        channel_layout.setHorizontalSpacing(8)
        channel_layout.setVerticalSpacing(3)
        self.channel_checks: dict[str, QCheckBox] = {}
        for index, channel in enumerate(ACTIVE_CHANNELS):
            checkbox = QCheckBox(channel)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._channel_selection_changed)
            self.channel_checks[channel] = checkbox
            channel_layout.addWidget(checkbox, index // 2, index % 2)
        reference_label = QLabel("REF:  Cz")
        reference_label.setObjectName("metadataLabel")
        ground_label = QLabel("GND:  Pz")
        ground_label.setObjectName("metadataLabel")
        channel_layout.addWidget(reference_label, 4, 0)
        channel_layout.addWidget(ground_label, 4, 1)
        layout.addWidget(channel_group)

        conditions_group = QGroupBox("ELECTRODE && SIGNAL CONDITIONS")
        conditions_layout = QGridLayout(conditions_group)
        conditions_layout.setContentsMargins(8, 7, 8, 8)
        conditions_layout.setHorizontalSpacing(6)
        conditions_layout.setVerticalSpacing(4)
        contact_labels = [state.value for state in ContactState]
        for index, channel in enumerate(ACTIVE_CHANNELS):
            channel_label = QLabel(channel)
            channel_label.setStyleSheet("color: #a7d9ed; font-weight: 800;")
            combo = QComboBox()
            combo.addItems(contact_labels)
            combo.setMinimumContentsLength(17)
            combo.currentTextChanged.connect(
                lambda text, selected_channel=channel: self._contact_condition_changed(
                    selected_channel, text
                )
            )
            self.condition_combos[channel] = combo
            conditions_layout.addWidget(channel_label, index, 0)
            conditions_layout.addWidget(combo, index, 1, 1, 2)

        conditions_layout.addWidget(QLabel("Signal artefact"), 8, 0)
        self.interference_combo = QComboBox()
        self.interference_combo.addItems([mode.value for mode in InterferenceMode])
        self.interference_combo.currentTextChanged.connect(self._interference_changed)
        conditions_layout.addWidget(self.interference_combo, 8, 1)
        self.reset_conditions_button = QPushButton("Reset")
        self.reset_conditions_button.clicked.connect(self.reset_conditions)
        conditions_layout.addWidget(self.reset_conditions_button, 8, 2)
        conditions_note = QLabel(
            "Software effect preview only. Real impedance / lead-off needs switched analogue hardware."
        )
        conditions_note.setWordWrap(True)
        conditions_note.setObjectName("panelNote")
        conditions_layout.addWidget(conditions_note, 9, 0, 1, 3)
        layout.addWidget(conditions_group)

        playback_group = QGroupBox("PLAYBACK CONFIGURATION")
        playback_layout = QVBoxLayout(playback_group)
        playback_layout.setContentsMargins(8, 7, 8, 8)
        playback_layout.setSpacing(4)
        playback_form = QFormLayout()
        playback_form.setHorizontalSpacing(8)
        playback_form.setVerticalSpacing(3)
        self.amplitude_combo = QComboBox()
        self.amplitude_combo.addItems(["0.25x", "0.5x", "1.0x", "2.0x"])
        self.amplitude_combo.setCurrentText("1.0x")
        self.amplitude_combo.currentTextChanged.connect(self._configuration_changed)
        self.playback_rate_label = QLabel("-")
        self.loop_checkbox = QCheckBox("Loop playback")
        self.loop_checkbox.toggled.connect(self._configuration_changed)
        playback_form.addRow("Amplitude scale", self.amplitude_combo)
        playback_form.addRow("Sample rate", self.playback_rate_label)
        playback_layout.addLayout(playback_form)
        playback_layout.addWidget(self.loop_checkbox)

        button_grid = QGridLayout()
        button_grid.setHorizontalSpacing(6)
        button_grid.setVerticalSpacing(4)
        self.start_button = QPushButton("START")
        self.start_button.setObjectName("startButton")
        self.pause_button = QPushButton("PAUSE")
        self.stop_button = QPushButton("STOP")
        self.reset_button = QPushButton("RESET")
        self.start_button.clicked.connect(self.start_playback)
        self.pause_button.clicked.connect(self.pause_playback)
        self.stop_button.clicked.connect(self.stop_playback)
        self.reset_button.clicked.connect(self.reset_playback)
        button_grid.addWidget(self.start_button, 0, 0)
        button_grid.addWidget(self.pause_button, 0, 1)
        button_grid.addWidget(self.stop_button, 1, 0)
        button_grid.addWidget(self.reset_button, 1, 1)
        playback_layout.addLayout(button_grid)
        layout.addWidget(playback_group)

        demo_buttons = QHBoxLayout()
        signal_chain_button = QPushButton("SIGNAL CHAIN")
        signal_chain_button.setObjectName("signalChainButton")
        signal_chain_button.clicked.connect(self.show_signal_chain)
        self.validation_button = QPushButton("RUN VALIDATION")
        self.validation_button.setObjectName("validationButton")
        self.validation_button.clicked.connect(self.run_validation)
        demo_buttons.addWidget(signal_chain_button)
        demo_buttons.addWidget(self.validation_button)
        layout.addLayout(demo_buttons)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(380)
        return scroll

    def _build_monitor_panel(self) -> QWidget:
        vertical = QSplitter(Qt.Orientation.Vertical)
        vertical.setChildrenCollapsible(False)

        waveform_group = QGroupBox("WAVEFORM MONITOR  |  SIMULATED EEG OUTPUT (µV)")
        waveform_layout = QVBoxLayout(waveform_group)
        self.event_banner = QLabel(
            "SYNTHETIC DEMO EVENT  |  20-30 s rhythmic test event  |  engineering demonstration only"
        )
        self.event_banner.setObjectName("eventBanner")
        self.event_banner.setVisible(False)
        waveform_layout.addWidget(self.event_banner)
        self.condition_banner = QLabel()
        self.condition_banner.setObjectName("conditionBanner")
        self.condition_banner.setVisible(False)
        waveform_layout.addWidget(self.condition_banner)
        self.graphics = pg.GraphicsLayoutWidget()
        self.graphics.setBackground("#0d1722")
        waveform_layout.addWidget(self.graphics, stretch=1)
        self._create_empty_plots()
        vertical.addWidget(waveform_group)

        lower = QSplitter(Qt.Orientation.Horizontal)
        lower.setChildrenCollapsible(False)
        lower.addWidget(self._build_dac_panel())
        lower.addWidget(self._build_log_panel())
        lower.setSizes([470, 590])
        vertical.addWidget(lower)
        vertical.setSizes([560, 280])
        return vertical

    def _build_dac_panel(self) -> QWidget:
        group = QGroupBox("VIRTUAL DAC MAPPING  |  SOFTWARE MODEL ONLY")
        layout = QVBoxLayout(group)
        note = QLabel(
            "16-bit virtual model  |  Target range: ±200 µV  |  Midscale represents 0 µV\n"
            "Software mapping only. Physical transfer depends on the chosen DAC and voltage "
            "reference, output topology, precision scaling network and calibration."
        )
        note.setWordWrap(True)
        note.setObjectName("panelNote")
        layout.addWidget(note)
        self.dac_table = QTableWidget(len(ACTIVE_CHANNELS), 3)
        self.dac_table.setHorizontalHeaderLabels(["Channel", "Target µV", "Mock DAC Code"])
        self.dac_table.verticalHeader().setVisible(False)
        self.dac_table.verticalHeader().setDefaultSectionSize(24)
        self.dac_table.verticalHeader().setMinimumSectionSize(22)
        self.dac_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.dac_table.horizontalHeader().setFixedHeight(28)
        self.dac_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.dac_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        for row, channel in enumerate(ACTIVE_CHANNELS):
            self.dac_table.setItem(row, 0, QTableWidgetItem(channel))
            self.dac_table.setItem(row, 1, QTableWidgetItem("+0.0"))
            self.dac_table.setItem(row, 2, QTableWidgetItem(str(self.dac.midscale_code)))
        layout.addWidget(self.dac_table)
        return group

    def _build_log_panel(self) -> QWidget:
        group = QGroupBox("COMMUNICATION LOG  |  DESKTOP ↔ MOCK DEVICE")
        layout = QVBoxLayout(group)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setPlaceholderText("Protocol traffic appears here after Connect...")
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.log.setFont(font)
        layout.addWidget(self.log)
        clear_button = QPushButton("Clear Log")
        clear_button.clicked.connect(self.log.clear)
        clear_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout.addWidget(clear_button, alignment=Qt.AlignmentFlag.AlignRight)
        return group

    def _create_empty_plots(self) -> None:
        self.graphics.clear()
        self.plot_items.clear()
        self.plot_curves.clear()
        self.cursor_lines.clear()
        self.event_regions.clear()
        linked_plot: pg.PlotItem | None = None
        for row, channel in enumerate(ACTIVE_CHANNELS):
            channel_label = self.graphics.addLabel(
                channel, row=row, col=0, color="#9fc4d8", size="10pt", bold=True
            )
            channel_label.setMaximumWidth(36)
            plot = self.graphics.addPlot(row=row, col=1)
            plot.showGrid(x=False, y=True, alpha=0.12)
            left_axis = plot.getAxis("left")
            left_axis.setWidth(42)
            plot.setMouseEnabled(x=True, y=False)
            plot.setMenuEnabled(False)
            plot.enableAutoRange(axis="y")
            if linked_plot is not None:
                plot.setXLink(linked_plot)
            else:
                linked_plot = plot
            plot.hideAxis("bottom")
            curve = plot.plot(pen=pg.mkPen("#6cd4ff", width=1.15), connect="finite")
            cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ffcb6b", width=1.3))
            plot.addItem(cursor)
            self.plot_items.append(plot)
            self.plot_curves[channel] = curve
            self.cursor_lines.append(cursor)
        self.graphics.ci.layout.setColumnFixedWidth(0, 38)
        for row in range(len(ACTIVE_CHANNELS)):
            self.graphics.ci.layout.setRowStretchFactor(row, 1)
        if linked_plot is not None:
            time_axis = pg.AxisItem(orientation="bottom")
            time_axis.setLabel("Time", units="s")
            time_axis.linkToView(linked_plot.getViewBox())
            time_axis.setHeight(34)
            self.graphics.addItem(time_axis, row=len(ACTIVE_CHANNELS), col=1)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #101820; color: #dce6ee; font-size: 10pt; }
            QMenuBar { background: #0d151c; color: #dce6ee; }
            QMenuBar::item:selected, QMenu { background: #1d2a34; }
            QGroupBox {
                border: 1px solid #314452; border-radius: 5px; margin-top: 10px;
                padding-top: 8px; font-weight: 700; color: #9fc4d8;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton {
                background: #253746; border: 1px solid #466174; border-radius: 4px;
                padding: 5px 8px; color: #ecf4f8;
            }
            QPushButton:hover { background: #304b5e; border-color: #67c7ec; }
            QPushButton:pressed { background: #172a36; }
            QPushButton:disabled { color: #6e7f89; background: #17222a; border-color: #283843; }
            QPushButton#startButton { background: #087f5b; border-color: #37b488; font-weight: 800; }
            QPushButton#startButton:hover { background: #0b956b; }
            QPushButton#signalChainButton, QPushButton#validationButton {
                background: #183c50; border-color: #4e849d; font-weight: 800;
            }
            QPushButton#validationButton { background: #164c42; border-color: #3f8f77; }
            QComboBox, QPlainTextEdit, QTableWidget {
                background: #0d1722; border: 1px solid #314452; border-radius: 3px;
                selection-background-color: #27526b;
            }
            QComboBox { padding: 5px; }
            QHeaderView::section { background: #1a2a36; color: #bdd5e1; padding: 6px; border: 0; }
            QTableWidget { gridline-color: #243844; }
            QLabel#appTitle { font-size: 21pt; font-weight: 800; color: #f2f8fb; }
            QLabel#subtitle { color: #8da5b4; }
            QLabel#modeBadge {
                background: #263948; color: #78dbff; border: 1px solid #3d6277;
                border-radius: 4px; padding: 7px 11px; font-weight: 800;
            }
            QLabel#eventBanner {
                background: #3b3020; color: #ffd27d; border: 1px solid #7c6236;
                border-radius: 3px; padding: 6px; font-weight: 700;
            }
            QLabel#conditionBanner {
                background: #2d2c35; color: #d8c9f0; border: 1px solid #665b78;
                border-radius: 3px; padding: 5px; font-weight: 700;
            }
            QLabel#metadataLabel { color: #a7d9ed; font-weight: 700; padding-top: 5px; }
            QLabel#deviceName { color: #ecf4f8; font-weight: 800; }
            QLabel#connectionState { font-weight: 800; }
            QLabel#panelNote, QLabel#footerNote { color: #8fa5b3; }
            QLabel#footerStatus { color: #b7d8e8; font-weight: 700; }
            QCheckBox { spacing: 7px; }
            QScrollBar:vertical { background: #101820; width: 10px; }
            QScrollBar::handle:vertical { background: #3b5362; min-height: 24px; border-radius: 4px; }
            QSplitter::handle { background: #1f313d; }
            QDialog { background: #0e1720; }
            QLabel#dialogTitle { font-size: 20pt; font-weight: 800; color: #f2f8fb; }
            QLabel#dialogSubtitle { color: #8fa9b8; font-size: 10pt; }
            QLabel#sectionTitle { font-size: 15pt; font-weight: 800; color: #dff5ff; }
            QLabel#sectionNote { color: #9bb2bf; }
            QLabel#arrowLabel { color: #61c8ec; font-size: 18pt; font-weight: 800; }
            QLabel#boundaryLabel {
                background: #403521; color: #ffd27d; border: 1px solid #7c6236;
                border-radius: 4px; padding: 7px; font-weight: 800;
            }
            QLabel#passBanner {
                background: #123f35; color: #7de3b8; border: 1px solid #347b65;
                border-radius: 5px; padding: 10px; font-size: 13pt; font-weight: 800;
            }
            QLabel#failBanner {
                background: #4b2528; color: #ffaaaa; border: 1px solid #934951;
                border-radius: 5px; padding: 10px; font-size: 13pt; font-weight: 800;
            }
            """
        )

    @property
    def amplitude_scale(self) -> float:
        return float(self.amplitude_combo.currentText().rstrip("x"))

    def _selected_channels(self) -> tuple[str, ...]:
        return tuple(channel for channel, box in self.channel_checks.items() if box.isChecked())

    def _append_log(self, message: str) -> None:
        stamp = QTime.currentTime().toString("HH:mm:ss.zzz")
        self.log.appendPlainText(f"{stamp}  {message}")

    def _guard(self, action: Callable[[], None], title: str) -> None:
        try:
            action()
        except Exception as exc:  # GUI boundary: convert implementation errors into clear feedback.
            QMessageBox.critical(self, title, str(exc))

    def connect_device(self) -> None:
        self._guard(self.device.connect, "Connection failed")
        self._update_device_status()
        self._update_controls()

    def disconnect_device(self) -> None:
        if self.playback.state is PlaybackState.RUNNING:
            self.playback.stop()
        self._guard(self.device.disconnect, "Disconnect failed")
        self._prepared_signature = None
        self._update_device_status()
        self._update_controls()

    def generate_synthetic(self) -> None:
        def action() -> None:
            data = generate_synthetic_eeg()
            output = data.save_npz(DEFAULT_DATA_PATH)
            self._set_data(data, output)
            self._append_log(f"APP   Generated synthetic dataset: {output.name}")

        self._guard(action, "Synthetic EEG generation failed")

    def load_eeg(self) -> None:
        start_dir = str(DEFAULT_DATA_PATH.parent if DEFAULT_DATA_PATH.parent.exists() else PROJECT_ROOT)
        filename, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load EEG test file",
            start_dir,
            "EEG files (*.npz *.edf);;EEG simulator NPZ (*.npz);;EDF files (*.edf)",
        )
        if not filename:
            return

        def action() -> None:
            self._set_data(load_eeg_file(filename), Path(filename))
            self._append_log(f"APP   Loaded EEG file: {Path(filename).name}")

        self._guard(action, "Could not load EEG")

    def _set_data(self, source: EEGData, path: Path) -> None:
        indices = []
        for channel in ACTIVE_CHANNELS:
            if channel not in source.channel_names:
                raise ValueError(f"EEG file is missing required channel {channel}")
            indices.append(source.channel_names.index(channel))
        self.data = EEGData(
            name=source.name,
            samples_uv=source.samples_uv[indices],
            sample_rate_hz=source.sample_rate_hz,
            channel_names=ACTIVE_CHANNELS,
            units=source.units,
            reference=source.reference,
            ground=source.ground,
            annotations=list(source.annotations),
            metadata=dict(source.metadata),
        )
        self.loaded_path = path
        self._conditioned_cache = None
        self._conditioned_cache_signature = None
        self.playback.configure(self.data.sample_rate_hz, self.data.sample_count, self.loop_checkbox.isChecked())
        self._prepared_signature = None
        self._update_file_info()
        self._draw_waveforms()
        self._set_cursor(0)
        self._update_dac_table(0)
        self._update_controls()

    def _update_file_info(self) -> None:
        if self.data is None or self.loaded_path is None:
            return
        self.file_name_label.setText(self.loaded_path.name)
        self.file_name_label.setToolTip(str(self.loaded_path))
        self.file_rate_label.setText(f"{self.data.sample_rate_hz:g} Hz")
        self.file_duration_label.setText(f"{self.data.duration_seconds:.1f} s")
        self.file_channels_label.setText(str(self.data.channel_count))
        self.file_samples_label.setText(f"{self.data.sample_count:,}")
        self.playback_rate_label.setText(f"{self.data.sample_rate_hz:g} Hz")

    def _conditioned_samples(self) -> np.ndarray:
        if self.data is None:
            raise RuntimeError("No EEG data loaded")
        signature = (id(self.data), self.amplitude_scale, self.conditions.signature)
        if self._conditioned_cache_signature != signature:
            scaled = self.data.samples_uv * self.amplitude_scale
            self._conditioned_cache = self.conditions.apply(
                scaled, self.data.sample_rate_hz
            )
            self._conditioned_cache_signature = signature
        assert self._conditioned_cache is not None
        return self._conditioned_cache

    def _draw_waveforms(self) -> None:
        if self.data is None:
            return
        scaled = self._conditioned_samples()
        time_s = self.data.time_seconds
        event_visible = False
        for index, channel in enumerate(ACTIVE_CHANNELS):
            curve = self.plot_curves[channel]
            curve.setData(time_s, scaled[index])
            curve.setVisible(self.channel_checks[channel].isChecked())
            state = self.conditions.contact_states[channel]
            if state is ContactState.LEAD_OFF:
                curve.setPen(pg.mkPen("#ff9b9b", width=1.25, style=Qt.PenStyle.DashLine))
            elif state in (ContactState.HIGH, ContactState.VERY_HIGH):
                curve.setPen(pg.mkPen("#ffc46b", width=1.2))
            elif state is ContactState.MODERATE:
                curve.setPen(pg.mkPen("#a6d8d0", width=1.15))
            else:
                curve.setPen(pg.mkPen("#6cd4ff", width=1.15))
            plot = self.plot_items[index]
            plot.setXRange(0.0, self.data.duration_seconds, padding=0.005)
            axis_limit = self._nice_axis_limit(scaled[index])
            plot.setYRange(-axis_limit * 1.12, axis_limit * 1.12, padding=0)
            plot.getAxis("left").setTicks(
                [[(0.0, f"±{axis_limit:g}")]]
            )
        for region in self.event_regions:
            region.scene().removeItem(region)
        self.event_regions.clear()
        for annotation in self.data.annotations:
            if "SYNTHETIC" not in annotation.description.upper():
                continue
            event_visible = True
            bounds = [annotation.onset_seconds, annotation.onset_seconds + annotation.duration_seconds]
            for plot in self.plot_items:
                region = pg.LinearRegionItem(
                    values=bounds,
                    movable=False,
                    brush=pg.mkBrush(QColor(255, 179, 71, 32)),
                    pen=pg.mkPen(QColor(255, 194, 102, 95), width=0.8),
                )
                region.setZValue(-10)
                plot.addItem(region)
                self.event_regions.append(region)
        self.event_banner.setVisible(event_visible)
        self._update_condition_banner()

    @staticmethod
    def _nice_axis_limit(values: np.ndarray) -> float:
        """Return a rounded symmetric display limit for one compact channel lane."""
        peak = float(np.max(np.abs(values)))
        if peak <= 0:
            return 1.0
        magnitude = 10.0 ** math.floor(math.log10(peak))
        normalised = peak / magnitude
        if normalised <= 1.0:
            nice_factor = 1.0
        elif normalised <= 2.0:
            nice_factor = 2.0
        elif normalised <= 5.0:
            nice_factor = 5.0
        else:
            nice_factor = 10.0
        return nice_factor * magnitude

    def _contact_condition_changed(self, channel: str, label: str) -> None:
        state = ContactState.from_label(label)
        previous = self.conditions.contact_states[channel]
        if state is previous:
            return
        self.conditions.set_contact_state(channel, state)
        self._style_condition_combo(channel, state)
        self._condition_settings_changed()
        self._append_log(
            f"APP   {channel} contact: {previous.short_label} → {state.short_label}"
        )

    def _interference_changed(self, label: str) -> None:
        mode = InterferenceMode.from_label(label)
        previous = self.conditions.interference_mode
        if mode is previous:
            return
        self.conditions.set_interference_mode(mode)
        if mode is InterferenceMode.NONE:
            self.interference_combo.setStyleSheet("")
        else:
            self.interference_combo.setStyleSheet(
                "QComboBox { background: #383126; color: #ffd28a; border-color: #8c6b38; }"
            )
        self._condition_settings_changed()
        self._append_log(
            f"APP   Signal artefact: {previous.value} → {mode.value} (all channels)"
        )

    def reset_conditions(self) -> None:
        had_conditions = bool(self.conditions.abnormal_summary)
        self.conditions.reset()
        for channel, combo in self.condition_combos.items():
            combo.blockSignals(True)
            combo.setCurrentText(ContactState.NORMAL.value)
            combo.blockSignals(False)
            self._style_condition_combo(channel, ContactState.NORMAL)
        self.interference_combo.blockSignals(True)
        self.interference_combo.setCurrentText(InterferenceMode.NONE.value)
        self.interference_combo.blockSignals(False)
        self.interference_combo.setStyleSheet("")
        self._condition_settings_changed()
        if had_conditions:
            self._append_log("APP   Electrode and signal conditions reset to normal")

    def _condition_settings_changed(self) -> None:
        self._conditioned_cache = None
        self._conditioned_cache_signature = None
        self._prepared_signature = None
        if self.data is not None:
            self._draw_waveforms()
            self._update_dac_table(self.playback.snapshot().sample_index)
        else:
            self._update_condition_banner()

    def _update_condition_banner(self) -> None:
        summary = self.conditions.abnormal_summary
        self.condition_banner.setVisible(bool(summary))
        if summary:
            self.condition_banner.setText(
                "SIMULATED CONDITIONS  |  " + "  ·  ".join(summary) + "  |  software preview"
            )

    def _style_condition_combo(self, channel: str, state: ContactState) -> None:
        combo = self.condition_combos[channel]
        if state is ContactState.LEAD_OFF:
            combo.setStyleSheet(
                "QComboBox { background: #44282b; color: #ffb0b0; border-color: #8d4d55; }"
            )
        elif state in (ContactState.HIGH, ContactState.VERY_HIGH):
            combo.setStyleSheet(
                "QComboBox { background: #3b3020; color: #ffd28a; border-color: #80643a; }"
            )
        elif state is ContactState.MODERATE:
            combo.setStyleSheet(
                "QComboBox { background: #23383a; color: #b8ded7; border-color: #4b7472; }"
            )
        else:
            combo.setStyleSheet("")

    def _channel_selection_changed(self) -> None:
        if not self._selected_channels():
            sender = self.sender()
            if isinstance(sender, QCheckBox):
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
            QMessageBox.information(self, "Channel selection", "At least one channel must remain active.")
            return
        for channel, checkbox in self.channel_checks.items():
            if channel in self.plot_curves:
                self.plot_curves[channel].setVisible(checkbox.isChecked())
        self._configuration_changed()

    def _configuration_changed(self) -> None:
        self._prepared_signature = None
        if self.data is not None:
            self.playback.loop = self.loop_checkbox.isChecked()
            self._draw_waveforms()
            self._update_dac_table(self.playback.snapshot().sample_index)

    def _configuration_signature(self) -> tuple[object, ...]:
        if self.data is None:
            return ()
        return (
            id(self.data),
            self.data.sample_rate_hz,
            self.amplitude_scale,
            self._selected_channels(),
            self.loop_checkbox.isChecked(),
            self.conditions.signature,
        )

    def start_playback(self) -> None:
        def action() -> None:
            if self.data is None:
                raise RuntimeError("Generate or load an EEG file first")
            status = self.device.get_status()
            if not status.connected:
                raise RuntimeError("Connect the Mock EEG Simulator first")

            if status.state is DeviceState.PAUSED and self._prepared_signature == self._configuration_signature():
                self.device.start()
                self.playback.start()
            else:
                selected = self._selected_channels()
                selected_indices = [ACTIVE_CHANNELS.index(channel) for channel in selected]
                self.device.configure(
                    self.data.sample_rate_hz,
                    self.amplitude_scale,
                    selected,
                    self.loop_checkbox.isChecked(),
                )
                self.device.upload_waveform(
                    self._conditioned_samples()[selected_indices]
                )
                self.playback.configure(
                    self.data.sample_rate_hz,
                    self.data.sample_count,
                    self.loop_checkbox.isChecked(),
                )
                self._prepared_signature = self._configuration_signature()
                self.device.start()
                self.playback.start()
            self._update_device_status()
            self._update_controls()

        self._guard(action, "Playback could not start")

    def pause_playback(self) -> None:
        def action() -> None:
            self.device.pause()
            self.playback.pause()
            self._update_device_status()
            self._update_controls()

        self._guard(action, "Playback could not pause")

    def stop_playback(self) -> None:
        def action() -> None:
            self.device.stop()
            self.playback.stop()
            self._set_cursor(0)
            self._update_dac_table(0)
            self._update_device_status()
            self._update_controls()

        self._guard(action, "Playback could not stop")

    def reset_playback(self) -> None:
        def action() -> None:
            self.device.reset()
            self.playback.reset()
            self._prepared_signature = None
            self._set_cursor(0)
            self._update_dac_table(0)
            self._update_device_status()
            self._update_controls()

        self._guard(action, "Device reset failed")

    def _refresh_playback(self) -> None:
        if self.data is None:
            return
        snapshot = self.playback.snapshot()
        self._set_cursor(snapshot.sample_index)
        self._update_dac_table(snapshot.sample_index)
        if snapshot.completed:
            try:
                self.device.stop()
            except RuntimeError:
                pass
            self._update_device_status()
            self._update_controls()

    def _set_cursor(self, sample_index: int) -> None:
        if self.data is None:
            return
        position = sample_index / self.data.sample_rate_hz
        for cursor in self.cursor_lines:
            cursor.setPos(position)
        self.position_label.setText(
            f"Position  {self._format_time(position)} / {self._format_time(self.data.duration_seconds)}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        remaining = seconds - minutes * 60
        return f"{minutes:02d}:{remaining:06.3f}"

    def _update_dac_table(self, sample_index: int) -> None:
        if self.data is None:
            return
        index = min(max(sample_index, 0), self.data.sample_count - 1)
        values = self._conditioned_samples()[:, index]
        codes = np.asarray(self.dac.microvolts_to_code(values))
        for row, channel in enumerate(ACTIVE_CHANNELS):
            enabled = self.channel_checks[channel].isChecked()
            value_text = f"{values[row]:+7.2f}" if enabled else "--"
            code_text = str(int(codes[row])) if enabled else "--"
            self.dac_table.item(row, 1).setText(value_text)
            self.dac_table.item(row, 2).setText(code_text)
            colour = QColor("#dce6ee" if enabled else "#687984")
            for column in range(3):
                self.dac_table.item(row, column).setForeground(colour)

    def _update_device_status(self) -> None:
        status = self.device.get_status()
        self.connection_label.setText("● Connected" if status.connected else "● Disconnected")
        self.connection_label.setStyleSheet(
            "color: #63d9a4;" if status.connected else "color: #ff8d8d;"
        )
        self.device_state_label.setText(status.state.value)

    def _update_controls(self) -> None:
        status = self.device.get_status()
        has_data = self.data is not None
        self.connect_button.setEnabled(not status.connected)
        self.disconnect_button.setEnabled(status.connected)
        self.start_button.setEnabled(
            has_data and status.connected and status.state is not DeviceState.RUNNING
        )
        self.pause_button.setEnabled(status.state is DeviceState.RUNNING)
        self.stop_button.setEnabled(status.state in (DeviceState.RUNNING, DeviceState.PAUSED))
        self.reset_button.setEnabled(status.connected)
        editing_enabled = status.state is not DeviceState.RUNNING
        self.generate_button.setEnabled(editing_enabled)
        self.load_button.setEnabled(editing_enabled)
        self.amplitude_combo.setEnabled(editing_enabled)
        self.loop_checkbox.setEnabled(editing_enabled)
        for checkbox in self.channel_checks.values():
            checkbox.setEnabled(editing_enabled)
        for combo in self.condition_combos.values():
            combo.setEnabled(editing_enabled)
        self.interference_combo.setEnabled(editing_enabled)
        self.reset_conditions_button.setEnabled(editing_enabled)
        self.validation_button.setEnabled(has_data)

    def show_signal_chain(self) -> None:
        self._create_signal_chain_dialog().exec()

    def _create_signal_chain_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("Signal Chain | End-to-End Engineering View")
        dialog.setObjectName("signalChainDialog")
        available = self.screen().availableGeometry()
        dialog.resize(min(1720, int(available.width() * 0.94)), min(980, int(available.height() * 0.94)))
        dialog.setMinimumSize(1180, 760)
        dialog.setStyleSheet(
            """
            QDialog#signalChainDialog, QDialog#signalChainDialog QWidget {
                background: #f3f6f8; color: #263842; font-size: 9.5pt;
            }
            QDialog#signalChainDialog QLabel { background: transparent; }
            QDialog#signalChainDialog QLabel#architectureTitle {
                color: #20343f; font-size: 18pt; font-weight: 800;
            }
            QDialog#signalChainDialog QLabel#architectureSubtitle {
                color: #58707d; font-size: 10pt;
            }
            QDialog#signalChainDialog QLabel#architectureSection {
                color: #314a57; font-size: 11pt; font-weight: 800;
            }
            QDialog#signalChainDialog QLabel#architectureArrow {
                color: #6b7f89; font-size: 17pt; font-weight: 700;
            }
            QDialog#signalChainDialog QPushButton {
                background: #ffffff; color: #314a57; border: 1px solid #aebdc5;
                border-radius: 5px; padding: 6px 18px;
            }
            QDialog#signalChainDialog QPushButton:hover { background: #e7eef2; }
            """
        )
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 12, 18, 10)
        layout.setSpacing(4)

        title = QLabel("SIGNAL CHAIN / SYSTEM ARCHITECTURE")
        title.setObjectName("architectureTitle")
        layout.addWidget(title)
        subtitle = QLabel("EEG Simulator → acquisition device → Automated Verification")
        subtitle.setObjectName("architectureSubtitle")
        layout.addWidget(subtitle)

        purpose = QFrame()
        purpose.setObjectName("purposeFrame")
        purpose.setStyleSheet(
            "QFrame#purposeFrame { background: #ffffff; border: 1px solid #d4dee3; "
            "border-radius: 6px; }"
        )
        purpose_layout = QVBoxLayout(purpose)
        purpose_layout.setContentsMargins(12, 3, 12, 3)
        purpose_layout.setSpacing(1)
        purpose_main = QLabel("Known Input → Controlled Physical Signal → acquisition device → Measured Result")
        purpose_main.setAlignment(Qt.AlignmentFlag.AlignCenter)
        purpose_main.setStyleSheet("color: #2f5262; font-weight: 800; font-size: 11pt;")
        purpose_sub = QLabel("Repeatable regression testing, fault injection and root-cause isolation")
        purpose_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        purpose_sub.setStyleSheet("color: #6b7f89;")
        purpose_layout.addWidget(purpose_main)
        purpose_layout.addWidget(purpose_sub)
        layout.addWidget(purpose)

        legend = QHBoxLayout()
        legend.setSpacing(6)
        legend.addWidget(self._architecture_badge("Implemented", "#4d8f82", "#edf7f3"))
        legend.addWidget(self._architecture_badge("Future Hardware", "#aa7935", "#fff8eb"))
        legend.addWidget(self._architecture_badge("Conceptual Acquisition Path", "#5684a5", "#eef5fa"))
        legend.addWidget(self._architecture_badge("Mock Validation", "#776a9b", "#f4f1f8"))
        legend.addStretch(1)
        layout.addLayout(legend)

        implemented = ("#4d8f82", "#edf7f3", "#a9cec5")
        future = ("#aa7935", "#fff8eb", "#dec89f")
        acquisition_palette = ("#5684a5", "#eef5fa", "#b4cad8")
        validation = ("#776a9b", "#f4f1f8", "#c9c0db")

        first_row = QHBoxLayout()
        first_row.setSpacing(8)
        first_row.addWidget(
            self._architecture_zone(
                "DESKTOP",
                (
                    ("EEG SOURCE", "Recorded EEG\nor Synthetic EEG", "8 channels · 256 Hz example"),
                    (
                        "DESKTOP\nSOFTWARE",
                        "Load / Parse\nChannels / Metadata\nPlayback Control",
                        "Amplitude · Loop · File · Fault Scenario",
                    ),
                ),
                implemented,
            ),
            stretch=2,
        )
        first_row.addWidget(self._architecture_arrow(), alignment=Qt.AlignmentFlag.AlignCenter)
        first_row.addWidget(
            self._architecture_zone(
                "FUTURE EEG SIMULATOR · EMBEDDED CONTROL",
                (
                    ("USB / SERIAL", "Commands\nWaveform Data\nStatus", ""),
                    ("MCU", "Ring Buffer\nState / Control", "Deterministic Playback"),
                    ("TIMER / DMA", "Precise Sample Timing\nData Transfer", ""),
                ),
                future,
            ),
            stretch=3,
        )
        layout.addLayout(first_row)

        continuation = QLabel("DETERMINISTIC PLAYBACK  ↓  SIGNAL GENERATION / ACQUISITION INPUT")
        continuation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        continuation.setStyleSheet("color: #6b7f89; font-weight: 800; padding: 1px;")
        layout.addWidget(continuation)

        analogue_row = QHBoxLayout()
        analogue_row.setSpacing(8)
        analogue_row.addWidget(
            self._architecture_zone(
                "FUTURE EEG SIMULATOR · ANALOGUE OUTPUT",
                (
                    (
                        "MULTI-CHANNEL\nDAC",
                        "Digital Code\n→ Electrical Output",
                        "Reference · Resolution · Sync",
                    ),
                    (
                        "LOW-NOISE\nANALOGUE",
                        "Scaling\nAttenuation\nBuffering",
                        "Calibrated µV Output",
                    ),
                    (
                        "ELECTRODE /\nFAULT",
                        "Impedance States\nLead Off\nMains Injection",
                        "Future switches / resistance",
                    ),
                    (
                        "ACQUISITION\nINTERFACE",
                        "F3 F4 C3 C4\nT3 T4 O1 O2",
                        "REF = Cz · GND = Pz",
                    ),
                ),
                future,
            ),
            stretch=4,
        )
        analogue_row.addWidget(self._architecture_arrow(), alignment=Qt.AlignmentFlag.AlignCenter)
        analogue_row.addWidget(
            self._architecture_zone(
                "ACQUISITION DEVICE",
                (
                    ("INPUT PROTECTION", "Protection\nInput Network", ""),
                    ("AFE / FILTERING", "Low-noise Front End\nSignal Conditioning", "Gain / Filtering"),
                    ("ADC", "µV Analogue\n→ Digital EEG", ""),
                ),
                acquisition_palette,
            ),
            stretch=3,
        )
        layout.addLayout(analogue_row)

        digital_continuation = QLabel("DIGITISED EEG  ↓  PROCESSING / RESULT")
        digital_continuation.setAlignment(Qt.AlignmentFlag.AlignCenter)
        digital_continuation.setStyleSheet(
            "color: #6b7f89; font-weight: 800; padding: 1px;"
        )
        layout.addWidget(digital_continuation)

        layout.addWidget(
            self._architecture_zone(
                "ACQUISITION · DIGITAL / OUTPUT",
                (
                    (
                        "MCU / DIGITAL PROCESSING",
                        "Buffering\nPreprocessing\nSystem Control",
                        "",
                    ),
                    ("EDGE AI", "AI Co-processor /\nAccelerator", "Local Inference"),
                    (
                        "WIRELESS / PORTAL",
                        "Wireless Transport\nCaptured Output",
                        "Expected vs Observed",
                    ),
                    ("DETECTION RESULT", "Event / Seizure Detection\nLatency / Status", ""),
                ),
                acquisition_palette,
            )
        )

        verification_heading = QLabel("THREE-LEVEL AUTOMATED VERIFICATION")
        verification_heading.setObjectName("architectureSection")
        layout.addWidget(verification_heading)
        verification_row = QHBoxLayout()
        verification_row.setSpacing(9)
        verification_row.addWidget(
            self._verification_card(
                "CHECK 1\nSIMULATOR OUTPUT",
                "Target EEG  vs  Measured Simulator Output",
                "Amplitude · RMSE · Correlation · Timing · Noise",
                "Is the simulator reproducing the target correctly?",
                "Mock Measurement",
                validation,
            ),
            stretch=1,
        )
        verification_row.addWidget(
            self._verification_card(
                "CHECK 2\nACQUISITION",
                "Target EEG  vs  acquired EEG",
                "RMSE · Correlation · Amplitude Ratio · Timing Offset",
                "Did acquisition device acquire the injected waveform correctly?",
                "Mock Acquisition Capture",
                validation,
            ),
            stretch=1,
        )
        verification_row.addWidget(
            self._verification_card(
                "CHECK 3\nDETECTION",
                "Known Event  vs  Detection Result",
                "Detected / Missed · False Alarm · Latency",
                "Did the expected event trigger the expected detection?",
                "Mock Detection",
                validation,
            ),
            stretch=1,
        )
        layout.addLayout(verification_row)

        root_cause = QFrame()
        root_cause.setObjectName("rootCauseFrame")
        root_cause.setStyleSheet(
            "QFrame#rootCauseFrame { background: #ffffff; border: 1px solid #c9c0db; "
            "border-radius: 6px; }"
        )
        root_layout = QVBoxLayout(root_cause)
        root_layout.setContentsMargins(10, 6, 10, 6)
        root_layout.setSpacing(3)
        root_title = QLabel("ROOT-CAUSE ISOLATION")
        root_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_title.setStyleSheet("color: #665889; font-weight: 800;")
        root_layout.addWidget(root_title)
        root_items = QHBoxLayout()
        root_items.setSpacing(8)
        for text in (
            "CHECK 1 FAIL  →  Simulator / analogue path",
            "CHECK 1 PASS + CHECK 2 FAIL  →  Acquisition / interface",
            "CHECK 1 + CHECK 2 PASS + CHECK 3 FAIL  →  Processing / detection",
        ):
            item = QLabel(text)
            item.setAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setWordWrap(True)
            item.setStyleSheet(
                "background: #f7f5fa; color: #4e465f; border-radius: 4px; padding: 5px;"
            )
            root_items.addWidget(item, stretch=1)
        root_layout.addLayout(root_items)
        layout.addWidget(root_cause)

        notes = QLabel(
            "Exact REF/GND electrical behaviour would be confirmed from the acquisition device schematic/interface specification.\n"
            "acquisition device path is conceptual; exact internal hardware and AI architecture are not assumed."
        )
        notes.setAlignment(Qt.AlignmentFlag.AlignCenter)
        notes.setStyleSheet("color: #687d88; font-size: 8.5pt;")
        layout.addWidget(notes)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        return dialog

    def _architecture_zone(
        self,
        title: str,
        cards: tuple[tuple[str, str, str], ...],
        palette: tuple[str, str, str],
    ) -> QWidget:
        zone = QWidget()
        zone_layout = QVBoxLayout(zone)
        zone_layout.setContentsMargins(0, 0, 0, 0)
        zone_layout.setSpacing(3)
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet(
            f"color: {palette[0]}; background: {palette[1]}; border-radius: 4px; "
            "padding: 3px; font-weight: 800;"
        )
        zone_layout.addWidget(heading)
        flow = QHBoxLayout()
        flow.setSpacing(5)
        for index, (card_title, body, footer) in enumerate(cards):
            flow.addWidget(
                self._architecture_card(card_title, body, footer, palette),
                stretch=1,
            )
            if index < len(cards) - 1:
                flow.addWidget(self._architecture_arrow(), alignment=Qt.AlignmentFlag.AlignCenter)
        zone_layout.addLayout(flow)
        return zone

    @staticmethod
    def _architecture_card(
        title: str,
        body: str,
        footer: str,
        palette: tuple[str, str, str],
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("architectureCard")
        card.setStyleSheet(
            f"QFrame#architectureCard {{ background: {palette[1]}; border: 1px solid {palette[2]}; "
            f"border-top: 3px solid {palette[0]}; border-radius: 6px; }}"
        )
        card.setMinimumHeight(96)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(7, 7, 7, 6)
        card_layout.setSpacing(3)
        heading = QLabel(title)
        heading.setWordWrap(True)
        heading.setStyleSheet(f"color: {palette[0]}; font-weight: 800; font-size: 8.8pt;")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body_label = QLabel(body)
        body_label.setWordWrap(True)
        body_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body_label.setStyleSheet("color: #2c3f49; font-size: 8.7pt;")
        card_layout.addWidget(heading)
        card_layout.addWidget(body_label, stretch=1)
        footer_label = QLabel(footer or " ")
        footer_label.setWordWrap(True)
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_label.setStyleSheet("color: #70828b; font-size: 7.5pt;")
        card_layout.addWidget(footer_label)
        return card

    @staticmethod
    def _architecture_arrow() -> QLabel:
        arrow = QLabel("→")
        arrow.setObjectName("architectureArrow")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow.setFixedWidth(16)
        return arrow

    @staticmethod
    def _architecture_badge(text: str, colour: str, fill: str) -> QLabel:
        badge = QLabel(text)
        badge.setStyleSheet(
            f"background: {fill}; color: {colour}; border: 1px solid {colour}; "
            "border-radius: 4px; padding: 3px 8px; font-weight: 700; font-size: 8pt;"
        )
        return badge

    @staticmethod
    def _verification_card(
        title: str,
        comparison: str,
        metrics: str,
        purpose: str,
        mock_label: str,
        palette: tuple[str, str, str],
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("verificationCard")
        card.setStyleSheet(
            f"QFrame#verificationCard {{ background: {palette[1]}; border: 1px solid {palette[2]}; "
            f"border-top: 3px solid {palette[0]}; border-radius: 6px; }}"
        )
        card.setMinimumHeight(112)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 7, 10, 7)
        card_layout.setSpacing(3)
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet(f"color: {palette[0]}; font-weight: 800;")
        comparison_label = QLabel(comparison)
        comparison_label.setWordWrap(True)
        comparison_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        comparison_label.setStyleSheet("color: #293c46; font-weight: 700;")
        metrics_label = QLabel(metrics)
        metrics_label.setWordWrap(True)
        metrics_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        metrics_label.setStyleSheet("color: #5d6f79; font-size: 8.5pt;")
        purpose_label = QLabel(purpose)
        purpose_label.setWordWrap(True)
        purpose_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        purpose_label.setStyleSheet("color: #687983; font-size: 8pt;")
        mock = QLabel(mock_label)
        mock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mock.setStyleSheet(
            f"color: {palette[0]}; background: #ffffff; border: 1px solid {palette[2]}; "
            "border-radius: 3px; padding: 2px; font-size: 7.8pt; font-weight: 700;"
        )
        card_layout.addWidget(heading)
        card_layout.addWidget(comparison_label)
        card_layout.addWidget(metrics_label)
        card_layout.addWidget(purpose_label)
        card_layout.addWidget(mock)
        return card

    def run_validation(self) -> None:
        def action() -> None:
            if self.data is None:
                raise RuntimeError("Generate or load the synthetic EEG dataset first")
            result = run_validation_suite(self.data)
            self._append_log(
                f"APP   Validation complete: {result.passed_count}/{result.total_count} PASS"
            )
            self._create_validation_dialog(result).exec()

        self._guard(action, "Validation could not run")

    def _create_validation_dialog(self, result: ValidationSuiteResult) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle("Three-Level Validation | Software Reference")
        dialog.resize(1040, 690)
        dialog.setMinimumSize(920, 620)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        title = QLabel("THREE-LEVEL VALIDATION RESULTS")
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "Deterministic synthetic reference · fixed seeds · explicit thresholds · software mocks only"
        )
        subtitle.setObjectName("dialogSubtitle")
        layout.addWidget(subtitle)
        if self.conditions.abnormal_summary:
            condition_note = QLabel(
                "Active fault preview is shown in the waveform monitor but is not scored by "
                "this fixed nominal validation suite."
            )
            condition_note.setWordWrap(True)
            condition_note.setStyleSheet(
                "background: #3b3020; color: #ffd28a; border: 1px solid #80643a; "
                "border-radius: 4px; padding: 5px;"
            )
            layout.addWidget(condition_note)

        banner = QLabel(
            f"{'PASS' if result.overall_passed else 'FAIL'}  ·  "
            f"{result.passed_count}/{result.total_count} verification levels passed"
        )
        banner.setObjectName("passBanner" if result.overall_passed else "failBanner")
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(banner)

        cards = QHBoxLayout()
        cards.setSpacing(10)
        cards.addWidget(self._waveform_result_card(result.simulator_output), stretch=1)
        cards.addWidget(self._acquisition_result_card(result.acquisition), stretch=1)
        cards.addWidget(self._detection_result_card(result.event_detection), stretch=1)
        layout.addLayout(cards, stretch=1)

        log_title = QLabel("VALIDATION LOG")
        log_title.setStyleSheet("color: #9fc4d8; font-weight: 800;")
        layout.addWidget(log_title)
        log = QPlainTextEdit("\n".join(result.log_lines))
        log.setReadOnly(True)
        log.setMaximumHeight(130)
        log.setFont(QFont("Consolas", 9))
        layout.addWidget(log)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        return dialog

    def _waveform_result_card(self, result: WaveformValidationResult) -> QFrame:
        threshold = DEFAULT_WAVEFORM_THRESHOLDS
        return self._result_card(
            "LEVEL 1 · SIMULATOR OUTPUT",
            result.passed,
            (
                ("RMSE", f"{result.rmse_uv:.3f} µV", f"≤ {threshold.max_rmse_uv:.2f} µV"),
                ("Correlation", f"{result.correlation:.5f}", f"≥ {threshold.min_correlation:.3f}"),
                ("Gain error", f"{result.gain_error_percent:.3f}%", f"≤ {threshold.max_gain_error_percent:.1f}%"),
                ("Offset", f"{result.offset_uv:+.3f} µV", f"|x| ≤ {threshold.max_abs_offset_uv:.2f} µV"),
            ),
            "Known target vs mock measured output",
        )

    def _acquisition_result_card(self, result: AcquisitionValidationResult) -> QFrame:
        threshold = DEFAULT_ACQUISITION_THRESHOLDS
        return self._result_card(
            "LEVEL 2 · ACQUISITION",
            result.passed,
            (
                ("Correlation", f"{result.correlation:.5f}", f"≥ {threshold.min_correlation:.3f}"),
                ("Amplitude ratio", f"{result.amplitude_ratio:.5f}", f"within ±{threshold.max_amplitude_ratio_error_percent:.1f}%"),
                ("Timing offset", f"{result.timing_offset_ms:.3f} ms", f"≤ {threshold.max_timing_offset_ms:.1f} ms"),
            ),
            "Target vs deterministic mock acquisition capture",
        )

    def _detection_result_card(self, result: DetectionValidationResult) -> QFrame:
        threshold = DEFAULT_DETECTION_THRESHOLDS
        return self._result_card(
            "LEVEL 3 · EVENT DETECTION",
            result.passed,
            (
                ("Known event", f"{result.expected_start_seconds:g}–{result.expected_end_seconds:g} s", "synthetic label"),
                ("Mock detected", f"{result.detected_onset_seconds:.2f} s", "fixed mock"),
                ("Latency", f"{result.latency_seconds:.2f} s", f"≤ {threshold.max_latency_seconds:.1f} s"),
                ("False alarm", "YES" if result.false_alarm else "NO", "required: NO"),
            ),
            "Known annotation vs mock detection result",
        )

    @staticmethod
    def _result_card(
        title: str,
        passed: bool,
        metrics: tuple[tuple[str, str, str], ...],
        note: str,
    ) -> QFrame:
        colour = "#5bd5a6" if passed else "#ff8d8d"
        card = QFrame()
        card.setObjectName("resultCard")
        card.setStyleSheet(
            f"QFrame#resultCard {{ background: #15232d; border: 1px solid #35505f; "
            f"border-top: 4px solid {colour}; border-radius: 6px; }}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(13, 12, 13, 12)
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet("color: #dff5ff; font-weight: 800;")
        status = QLabel("PASS" if passed else "FAIL")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setStyleSheet(f"color: {colour}; font-size: 18pt; font-weight: 900;")
        layout.addWidget(heading)
        layout.addWidget(status)
        grid = QGridLayout()
        grid.setVerticalSpacing(8)
        for row, (label, value, limit) in enumerate(metrics):
            name = QLabel(label)
            name.setStyleSheet("color: #90aab8;")
            measured = QLabel(value)
            measured.setStyleSheet("color: #edf7fb; font-weight: 800;")
            requirement = QLabel(limit)
            requirement.setAlignment(Qt.AlignmentFlag.AlignRight)
            requirement.setStyleSheet("color: #7f9aa8; font-size: 8pt;")
            grid.addWidget(name, row, 0)
            grid.addWidget(measured, row, 1)
            grid.addWidget(requirement, row, 2)
        layout.addLayout(grid)
        layout.addStretch(1)
        description = QLabel(note)
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setStyleSheet("color: #8fa6b4; font-size: 8pt;")
        layout.addWidget(description)
        return card

    def run_smoke_scenario(self) -> None:
        """Exercise the normal demo flow without dialogs for automated GUI validation."""
        self.generate_synthetic()
        self._set_data(load_eeg_file(DEFAULT_DATA_PATH), DEFAULT_DATA_PATH)
        self._append_log(f"APP   Reloaded generated NPZ: {DEFAULT_DATA_PATH.name}")
        self.connect_device()
        self.start_playback()

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.refresh_timer.stop()
        if self.device.get_status().connected:
            try:
                self.device.disconnect()
            except Exception:
                pass
        super().closeEvent(event)
