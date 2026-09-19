"""Reusable signal and virtual-DAC logic for the browser demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
    contact_states: Mapping[str, ContactState] | None = None,
    interference_mode: InterferenceMode = InterferenceMode.NONE,
    *,
    contact_channel: str | None = None,
    contact_state: ContactState = ContactState.NORMAL,
) -> DemoWaveform:
    """Generate the deterministic demo with independent per-channel conditions.

    The keyword-only contact_channel/contact_state pair is retained for
    compatibility with the first browser demo. The console uses contact_states
    so all eight electrode conditions can be inspected at once.
    """

    if amplitude_scale <= 0:
        raise ValueError("amplitude_scale must be positive")

    data = generate_synthetic_eeg()
    states = dict(contact_states or {})
    if contact_channel is not None:
        states[contact_channel] = contact_state

    unknown = sorted(set(states) - set(data.channel_names))
    if unknown:
        raise ValueError(f"Unknown channel(s): {', '.join(unknown)}")

    scaled = data.scaled(amplitude_scale)
    conditions = ElectrodeConditionModel(data.channel_names)
    for channel, state in states.items():
        conditions.set_contact_state(channel, state)
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


def format_time(seconds: float) -> str:
    """Format seconds as the compact clock used by both simulator front ends."""

    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:06.3f}"
