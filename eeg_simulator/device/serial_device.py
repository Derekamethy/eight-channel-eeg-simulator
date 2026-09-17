"""PySerial-backed skeleton for a future STM32 USB CDC connection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..protocol import encode_message
from .base import DeviceState, DeviceStatus, SimulatorDevice


class SerialSimulatorDevice(SimulatorDevice):
    """Future hardware adapter; not used by the desktop demo GUI.

    The text commands mirror MockSimulatorDevice. A real implementation would
    define framing, checksums, binary waveform transfer, timeouts, versioning,
    and recovery together with the STM32 firmware.
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0, log_callback=None):
        super().__init__(log_callback)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None
        self._state = DeviceState.DISCONNECTED

    def _send(self, message: str) -> str:
        if self._serial is None:
            raise RuntimeError("Serial device is not connected")
        self._log(f"TX > {message}")
        self._serial.write((message + "\n").encode("ascii"))
        response = self._serial.readline().decode("ascii", errors="replace").strip()
        self._log(f"RX < {response or '[timeout]'}")
        if not response:
            raise TimeoutError("No response from serial simulator device")
        return response

    def connect(self) -> None:
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Serial support requires the 'pyserial' package") from exc
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self._send("CONNECT")
        self._state = DeviceState.IDLE

    def disconnect(self) -> None:
        if self._serial is not None:
            try:
                self._send("DISCONNECT")
            finally:
                self._serial.close()
                self._serial = None
        self._state = DeviceState.DISCONNECTED

    def configure(
        self, sample_rate_hz: float, amplitude_scale: float, channels: tuple[str, ...], loop: bool
    ) -> None:
        self._send(
            encode_message(
                "CONFIG",
                fs=f"{sample_rate_hz:g}",
                amp=f"{amplitude_scale:g}",
                channels=",".join(channels),
                loop=int(loop),
            )
        )
        self._state = DeviceState.IDLE

    def upload_waveform(self, samples_uv: NDArray[np.float64]) -> None:
        # Deliberately a header-only skeleton: binary framing belongs in the
        # future desktop/firmware protocol specification.
        waveform = np.asarray(samples_uv)
        self._send(
            encode_message("LOAD_WAVEFORM", samples=waveform.shape[1], channels=waveform.shape[0])
        )
        raise NotImplementedError("Binary serial waveform transfer is future hardware work")

    def start(self) -> None:
        self._send("START")
        self._state = DeviceState.RUNNING

    def pause(self) -> None:
        self._send("PAUSE")
        self._state = DeviceState.PAUSED

    def stop(self) -> None:
        self._send("STOP")
        self._state = DeviceState.READY

    def reset(self) -> None:
        self._send("RESET")
        self._state = DeviceState.IDLE

    def get_status(self) -> DeviceStatus:
        return DeviceStatus(
            connected=self._serial is not None,
            state=self._state,
            mode="SERIAL",
            detail=f"{self.port} @ {self.baudrate} baud",
        )
