"""Persistent device/playback controller used by the browser console."""

from __future__ import annotations

from datetime import datetime
import time
from typing import Callable, Hashable

import numpy as np
from numpy.typing import NDArray

from eeg_simulator.device.base import DeviceState, DeviceStatus
from eeg_simulator.device.mock_device import MockSimulatorDevice
from eeg_simulator.playback import PlaybackClock, PlaybackSnapshot, PlaybackState


class WebSimulatorController:
    """Mirror the desktop device and playback state flow inside one web session."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.logs: list[str] = []
        self.device = MockSimulatorDevice(self._append_device_log)
        self.playback = PlaybackClock(clock)
        self.prepared_signature: Hashable | None = None

    def _stamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def _append_device_log(self, message: str) -> None:
        self.logs.append(f"{self._stamp()}  {message}")
        if len(self.logs) > 250:
            del self.logs[:-250]

    def append_app_log(self, message: str) -> None:
        self.logs.append(f"{self._stamp()}  APP   {message}")
        if len(self.logs) > 250:
            del self.logs[:-250]

    def clear_logs(self) -> None:
        self.logs.clear()

    @property
    def status(self) -> DeviceStatus:
        return self.device.get_status()

    def connect(self) -> None:
        self.device.connect()

    def disconnect(self) -> None:
        if self.playback.state is PlaybackState.RUNNING:
            self.playback.stop()
        self.device.disconnect()
        self.playback.stop()
        self.prepared_signature = None

    def start(
        self,
        samples_uv: NDArray[np.float64],
        sample_rate_hz: float,
        amplitude_scale: float,
        channels: tuple[str, ...],
        loop: bool,
        signature: Hashable,
    ) -> None:
        status = self.device.get_status()
        if not status.connected:
            raise RuntimeError("Connect the Mock EEG Simulator first")
        if not channels:
            raise ValueError("At least one channel must be active")

        waveform = np.asarray(samples_uv, dtype=np.float64)
        if waveform.ndim != 2 or waveform.shape[0] != len(channels):
            raise ValueError("Waveform dimensions do not match the selected channels")

        if status.state is DeviceState.PAUSED and self.prepared_signature == signature:
            self.device.start()
            self.playback.start()
            return

        self.device.configure(sample_rate_hz, amplitude_scale, channels, loop)
        self.device.upload_waveform(waveform)
        self.playback.configure(sample_rate_hz, waveform.shape[1], loop)
        self.prepared_signature = signature
        self.device.start()
        self.playback.start()

    def pause(self) -> None:
        self.device.pause()
        self.playback.pause()

    def stop(self) -> None:
        self.device.stop()
        self.playback.stop()

    def reset(self) -> None:
        self.device.reset()
        self.playback.reset()
        self.prepared_signature = None

    def snapshot(self) -> PlaybackSnapshot:
        snapshot = self.playback.snapshot()
        if snapshot.completed and self.device.get_status().state is DeviceState.RUNNING:
            self.device.stop()
        return snapshot
