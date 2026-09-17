"""Deterministic software approximations of electrode and signal conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ContactState(Enum):
    NORMAL = "Normal (5 kΩ)"
    MODERATE = "Moderate (20 kΩ)"
    HIGH = "High Z (50 kΩ)"
    VERY_HIGH = "Very High (100 kΩ)"
    LEAD_OFF = "Lead Off"

    @classmethod
    def from_label(cls, label: str) -> "ContactState":
        return next(state for state in cls if state.value == label)

    @property
    def short_label(self) -> str:
        return {
            ContactState.NORMAL: "Normal",
            ContactState.MODERATE: "Moderate Z",
            ContactState.HIGH: "High Z",
            ContactState.VERY_HIGH: "Very High Z",
            ContactState.LEAD_OFF: "LEAD OFF",
        }[self]


class InterferenceMode(Enum):
    NONE = "None"
    MAINS_50_HZ = "50 Hz mains (6 µV)"
    MAINS_60_HZ = "60 Hz mains (6 µV)"

    @classmethod
    def from_label(cls, label: str) -> "InterferenceMode":
        return next(mode for mode in cls if mode.value == label)

    @property
    def frequency_hz(self) -> float | None:
        return {
            InterferenceMode.NONE: None,
            InterferenceMode.MAINS_50_HZ: 50.0,
            InterferenceMode.MAINS_60_HZ: 60.0,
        }[self]


@dataclass
class ElectrodeConditionModel:
    """Channel conditions plus a fixed-seed signal-effect preview.

    The resulting waveform is a software demonstration. Real source impedance
    and lead-off states require switched analogue hardware after the DAC stage.
    """

    channel_names: tuple[str, ...]
    seed: int = 20260826
    contact_states: dict[str, ContactState] = field(init=False)
    interference_mode: InterferenceMode = field(
        default=InterferenceMode.NONE, init=False
    )

    def __post_init__(self) -> None:
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be non-empty and unique")
        self.contact_states = {
            channel: ContactState.NORMAL for channel in self.channel_names
        }

    def set_contact_state(self, channel: str, state: ContactState) -> None:
        if channel not in self.contact_states:
            raise ValueError(f"Unknown channel: {channel}")
        self.contact_states[channel] = state

    def set_interference_mode(self, mode: InterferenceMode) -> None:
        self.interference_mode = mode

    def reset(self) -> None:
        for channel in self.channel_names:
            self.contact_states[channel] = ContactState.NORMAL
        self.interference_mode = InterferenceMode.NONE

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            *(self.contact_states[channel].value for channel in self.channel_names),
            self.interference_mode.value,
        )

    @property
    def abnormal_summary(self) -> tuple[str, ...]:
        items = tuple(
            f"{channel}: {self.contact_states[channel].short_label}"
            for channel in self.channel_names
            if self.contact_states[channel] is not ContactState.NORMAL
        )
        if self.interference_mode is not InterferenceMode.NONE:
            items += (f"All: {self.interference_mode.value}",)
        return items

    def apply(
        self,
        samples_uv: NDArray[np.float64],
        sample_rate_hz: float,
    ) -> NDArray[np.float64]:
        samples = np.asarray(samples_uv, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] != len(self.channel_names):
            raise ValueError("samples_uv must match the configured channel count")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

        output = samples.copy()
        time_s = np.arange(samples.shape[1], dtype=np.float64) / sample_rate_hz
        for index, channel in enumerate(self.channel_names):
            state = self.contact_states[channel]
            if state is ContactState.NORMAL:
                continue
            rng = np.random.default_rng(self.seed + index)
            phase = index * 0.41
            if state is ContactState.LEAD_OFF:
                output[index] = (
                    15.0 * np.sin(2.0 * np.pi * 0.18 * time_s + phase)
                    + 4.0 * np.sin(2.0 * np.pi * 50.0 * time_s + phase)
                    + rng.normal(0.0, 7.0, samples.shape[1])
                )
                continue
            noise_uv, pickup_uv = {
                ContactState.MODERATE: (0.6, 1.0),
                ContactState.HIGH: (1.5, 3.0),
                ContactState.VERY_HIGH: (2.8, 6.0),
            }[state]
            output[index] += rng.normal(0.0, noise_uv, samples.shape[1])
            output[index] += pickup_uv * np.sin(
                2.0 * np.pi * 50.0 * time_s + phase
            )

        frequency_hz = self.interference_mode.frequency_hz
        if frequency_hz is not None:
            for index in range(len(self.channel_names)):
                output[index] += 6.0 * np.sin(
                    2.0 * np.pi * frequency_hz * time_s + index * 0.13
                )
        return output
