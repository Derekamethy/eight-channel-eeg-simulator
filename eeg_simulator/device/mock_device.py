"""Deterministic in-process stand-in for a future embedded EEG simulator."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..protocol import encode_message
from .base import DeviceState, DeviceStatus, SimulatorDevice


class MockSimulatorDevice(SimulatorDevice):
    def __init__(self, log_callback=None) -> None:
        super().__init__(log_callback)
        self._state = DeviceState.DISCONNECTED
        self._configuration: dict[str, object] = {}
        self._waveform: NDArray[np.float64] | None = None

    def _require_connected(self) -> None:
        if self._state is DeviceState.DISCONNECTED:
            raise RuntimeError("Mock device is not connected")

    def _exchange(self, command: str, response: str) -> None:
        self._log(f"TX > {command}")
        self._log(f"RX < {response}")

    def connect(self) -> None:
        if self._state is not DeviceState.DISCONNECTED:
            return
        self._exchange("CONNECT", "ACK CONNECT")
        self._state = DeviceState.IDLE

    def disconnect(self) -> None:
        if self._state is DeviceState.DISCONNECTED:
            return
        self._exchange("DISCONNECT", "ACK DISCONNECT")
        self._state = DeviceState.DISCONNECTED
        self._waveform = None
        self._configuration.clear()

    def configure(
        self, sample_rate_hz: float, amplitude_scale: float, channels: tuple[str, ...], loop: bool
    ) -> None:
        self._require_connected()
        if self._state is DeviceState.RUNNING:
            raise RuntimeError("Stop playback before reconfiguring the device")
        if sample_rate_hz <= 0 or amplitude_scale <= 0 or not channels:
            raise ValueError("Invalid simulator configuration")
        self._configuration = {
            "sample_rate_hz": sample_rate_hz,
            "amplitude_scale": amplitude_scale,
            "channels": channels,
            "loop": loop,
        }
        command = encode_message(
            "CONFIG",
            fs=f"{sample_rate_hz:g}",
            amp=f"{amplitude_scale:g}",
            channels=len(channels),
            loop=int(loop),
        )
        self._exchange(command, "ACK CONFIG")
        self._state = DeviceState.IDLE

    def upload_waveform(self, samples_uv: NDArray[np.float64]) -> None:
        self._require_connected()
        if not self._configuration:
            raise RuntimeError("Configure the mock device before uploading a waveform")
        waveform = np.asarray(samples_uv, dtype=np.float64)
        expected_channels = len(self._configuration["channels"])
        if waveform.ndim != 2 or waveform.shape[0] != expected_channels or waveform.shape[1] == 0:
            raise ValueError("Waveform dimensions do not match the active channel configuration")
        self._waveform = waveform.copy()
        command = encode_message(
            "LOAD_WAVEFORM", samples=waveform.shape[1], channels=waveform.shape[0]
        )
        self._exchange(command, "READY")
        self._state = DeviceState.READY

    def start(self) -> None:
        self._require_connected()
        if self._state not in (DeviceState.READY, DeviceState.PAUSED):
            raise RuntimeError("Upload a waveform before starting playback")
        self._exchange("START", "RUNNING")
        self._state = DeviceState.RUNNING

    def pause(self) -> None:
        self._require_connected()
        if self._state is not DeviceState.RUNNING:
            raise RuntimeError("The mock device is not running")
        self._exchange("PAUSE", "PAUSED")
        self._state = DeviceState.PAUSED

    def stop(self) -> None:
        self._require_connected()
        if self._waveform is None:
            self._state = DeviceState.IDLE
            return
        self._exchange("STOP", "STOPPED")
        self._state = DeviceState.READY

    def reset(self) -> None:
        self._require_connected()
        self._exchange("RESET", "ACK RESET")
        self._state = DeviceState.IDLE
        self._waveform = None
        self._configuration.clear()

    def get_status(self) -> DeviceStatus:
        return DeviceStatus(
            connected=self._state is not DeviceState.DISCONNECTED,
            state=self._state,
            mode="MOCK",
            detail="In-process device model",
        )
