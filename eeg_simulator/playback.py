"""Monotonic-clock playback model independent of GUI refresh timing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable


class PlaybackState(str, Enum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"


@dataclass
class PlaybackSnapshot:
    state: PlaybackState
    sample_index: int
    position_seconds: float
    completed: bool = False


class PlaybackClock:
    """Derive sample position from elapsed monotonic time, not timer tick count."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self.sample_rate_hz = 1.0
        self.sample_count = 1
        self.loop = False
        self.state = PlaybackState.STOPPED
        self._anchor_time = 0.0
        self._anchor_sample = 0

    def configure(self, sample_rate_hz: float, sample_count: int, loop: bool = False) -> None:
        if sample_rate_hz <= 0 or sample_count <= 0:
            raise ValueError("Playback configuration must be positive")
        self.sample_rate_hz = float(sample_rate_hz)
        self.sample_count = int(sample_count)
        self.loop = bool(loop)
        self.stop()

    def start(self) -> None:
        if self.state is PlaybackState.RUNNING:
            return
        self._anchor_time = self._clock()
        self.state = PlaybackState.RUNNING

    def pause(self) -> None:
        if self.state is PlaybackState.RUNNING:
            self._anchor_sample = self.snapshot().sample_index
            self.state = PlaybackState.PAUSED

    def stop(self) -> None:
        self.state = PlaybackState.STOPPED
        self._anchor_sample = 0

    def reset(self) -> None:
        self.stop()

    def snapshot(self) -> PlaybackSnapshot:
        if self.state is not PlaybackState.RUNNING:
            return PlaybackSnapshot(
                self.state,
                min(self._anchor_sample, self.sample_count - 1),
                min(self._anchor_sample, self.sample_count - 1) / self.sample_rate_hz,
            )

        elapsed_samples = int((self._clock() - self._anchor_time) * self.sample_rate_hz)
        absolute_index = self._anchor_sample + elapsed_samples
        if absolute_index < self.sample_count:
            index = absolute_index
            return PlaybackSnapshot(self.state, index, index / self.sample_rate_hz)
        if self.loop:
            index = absolute_index % self.sample_count
            return PlaybackSnapshot(self.state, index, index / self.sample_rate_hz)

        self.state = PlaybackState.STOPPED
        self._anchor_sample = self.sample_count - 1
        return PlaybackSnapshot(
            self.state,
            self.sample_count - 1,
            (self.sample_count - 1) / self.sample_rate_hz,
            completed=True,
        )
