"""Deterministic, plausible-looking synthetic EEG for engineering demonstrations."""

from __future__ import annotations

import numpy as np

from .eeg_data import EEGAnnotation, EEGData


ACTIVE_CHANNELS = ("F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2")


def generate_synthetic_eeg(
    sample_rate_hz: int = 256,
    duration_seconds: float = 60.0,
    seed: int = 20260814,
) -> EEGData:
    """Create repeatable, non-clinical signals in microvolts.

    The 20-30 second interval is deliberately conspicuous and labelled as a
    synthetic test event. It is not intended to model or classify a seizure.
    """
    if sample_rate_hz <= 0 or duration_seconds <= 0:
        raise ValueError("Sample rate and duration must be positive")

    sample_count = int(round(sample_rate_hz * duration_seconds))
    time_s = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    rng = np.random.default_rng(seed)
    signals = np.empty((len(ACTIVE_CHANNELS), sample_count), dtype=np.float64)

    event_start = 20.0
    event_end = min(30.0, duration_seconds)
    event_mask = (time_s >= event_start) & (time_s < event_end)
    event_envelope = np.zeros_like(time_s)
    if event_end > event_start:
        event_time = time_s[event_mask] - event_start
        event_duration = event_end - event_start
        edge_seconds = min(1.0, event_duration / 4.0)
        envelope = np.ones_like(event_time)
        fade_in = event_time < edge_seconds
        fade_out = event_time > event_duration - edge_seconds
        envelope[fade_in] = 0.5 - 0.5 * np.cos(np.pi * event_time[fade_in] / edge_seconds)
        envelope[fade_out] = 0.5 - 0.5 * np.cos(
            np.pi * (event_duration - event_time[fade_out]) / edge_seconds
        )
        event_envelope[event_mask] = envelope

    for index, _channel in enumerate(ACTIVE_CHANNELS):
        phase = index * 0.47
        slow_activity = (
            (8.0 + index * 0.35) * np.sin(2 * np.pi * (0.65 + index * 0.015) * time_s + phase)
            + (4.5 + index * 0.2) * np.sin(2 * np.pi * (1.7 + index * 0.035) * time_s + phase / 2)
            + 2.0 * np.sin(2 * np.pi * (5.5 + index * 0.08) * time_s + 0.2 * index)
        )
        white_noise = rng.normal(0.0, 2.8 + index * 0.1, sample_count)
        # A short moving average gives low-amplitude correlated background activity.
        coloured_noise = np.convolve(white_noise, np.ones(5) / 5.0, mode="same")
        burst_frequency = 3.3 + 0.08 * index
        burst = (
            (22.0 + index * 1.2)
            * np.sin(2 * np.pi * burst_frequency * time_s + phase)
            * event_envelope
        )
        signals[index] = slow_activity + coloured_noise + burst

    annotations = []
    if event_end > event_start:
        annotations.append(
            EEGAnnotation(
                onset_seconds=event_start,
                duration_seconds=event_end - event_start,
                description="SYNTHETIC DEMO EVENT - synthetic rhythmic test event",
            )
        )

    return EEGData(
        name="synthetic_eeg_demo",
        samples_uv=signals,
        sample_rate_hz=float(sample_rate_hz),
        channel_names=ACTIVE_CHANNELS,
        reference="Cz",
        ground="Pz",
        annotations=annotations,
        metadata={
            "synthetic": True,
            "clinical_validity": "None - engineering demonstration data only",
            "event_label": "20-30 s: synthetic rhythmic test event",
            "generator_seed": seed,
        },
    )
