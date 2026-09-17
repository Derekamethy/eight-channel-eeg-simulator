"""Hardware-independent simulator device contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class DeviceState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    IDLE = "IDLE"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"


@dataclass(frozen=True)
class DeviceStatus:
    connected: bool
    state: DeviceState
    mode: str
    detail: str = ""


LogCallback = Callable[[str], None]


class SimulatorDevice(ABC):
    """Interface used by the GUI, independent of USB/serial implementation."""

    def __init__(self, log_callback: LogCallback | None = None) -> None:
        self.log_callback = log_callback or (lambda _message: None)

    def _log(self, message: str) -> None:
        self.log_callback(message)

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def configure(
        self, sample_rate_hz: float, amplitude_scale: float, channels: tuple[str, ...], loop: bool
    ) -> None: ...

    @abstractmethod
    def upload_waveform(self, samples_uv: NDArray[np.float64]) -> None: ...

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_status(self) -> DeviceStatus: ...
