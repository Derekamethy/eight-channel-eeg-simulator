"""Reusable signal and virtual-DAC logic for the browser demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from eeg_simulator.dac_model import VirtualDAC
from eeg_simulator.fault_conditions import (
    ContactState,
    ElectrodeConditionModel,
    InterferenceMode,
)
from eeg_simulator.synthetic_eeg import generate_synthetic_eeg


@dataclass(frozen=True)
class DemoWaveform:
    """Prepared deterministic waveform used by the browser front end."""

    samples_uv: NDArray[np.float64]
    dac_codes: NDArray[np.uint32]
    channel_names: tuple[str, ...]
    sample_rate_hz: float
    duration_seconds: float
    annotations: tuple[tuple[float, float, str], ...]
    abnormal_summary: tuple[str, ...]


def build_demo_waveform(
    amplitude_scale: float = 1.0,
    contact_channel: str = "F3",
    contact_state: ContactState = ContactState.NORMAL,
    interference_mode: InterferenceMode = InterferenceMode.NONE,
) -> DemoWaveform:
    """Generate the same deterministic demo signal used by the desktop project."""

    if amplitude_scale <= 0:
        raise ValueError("amplitude_scale must be positive")

    data = generate_synthetic_eeg()
    if contact_channel not in data.channel_names:
        raise ValueError(f"Unknown channel: {contact_channel}")

    scaled = data.scaled(amplitude_scale)
    conditions = ElectrodeConditionModel(data.channel_names)
    conditions.set_contact_state(contact_channel, contact_state)
    conditions.set_interference_mode(interference_mode)
    conditioned = conditions.apply(scaled.samples_uv, scaled.sample_rate_hz)

    dac = VirtualDAC()
    dac_codes = np.asarray(dac.microvolts_to_code(conditioned), dtype=np.uint32)

    annotations = tuple(
        (item.onset_seconds, item.duration_seconds, item.description)
        for item in data.annotations
    )
    return DemoWaveform(
        samples_uv=conditioned,
        dac_codes=dac_codes,
        channel_names=data.channel_names,
        sample_rate_hz=data.sample_rate_hz,
        duration_seconds=data.duration_seconds,
        annotations=annotations,
        abnormal_summary=conditions.abnormal_summary,
    )


def time_to_sample_index(
    position_seconds: float,
    sample_rate_hz: float,
    sample_count: int,
) -> int:
    """Convert a UI cursor position to a safe waveform sample index."""

    if sample_rate_hz <= 0 or sample_count <= 0:
        raise ValueError("sample_rate_hz and sample_count must be positive")
    index = int(round(float(position_seconds) * float(sample_rate_hz)))
    return min(max(index, 0), sample_count - 1)
