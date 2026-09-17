"""Deterministic, software-only reference checks for the engineering demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .eeg_data import EEGAnnotation, EEGData


SIMULATOR_MEASUREMENT_SEED = 20260821
ACQUISITION_CAPTURE_SEED = 20260822
MOCK_DETECTED_ONSET_SECONDS = 20.75


@dataclass(frozen=True)
class WaveformThresholds:
    max_rmse_uv: float = 0.50
    min_correlation: float = 0.995
    max_gain_error_percent: float = 1.0
    max_abs_offset_uv: float = 0.50


@dataclass(frozen=True)
class AcquisitionThresholds:
    min_correlation: float = 0.990
    max_amplitude_ratio_error_percent: float = 3.0
    max_timing_offset_ms: float = 5.0


@dataclass(frozen=True)
class DetectionThresholds:
    max_latency_seconds: float = 1.0


DEFAULT_WAVEFORM_THRESHOLDS = WaveformThresholds()
DEFAULT_ACQUISITION_THRESHOLDS = AcquisitionThresholds()
DEFAULT_DETECTION_THRESHOLDS = DetectionThresholds()


@dataclass(frozen=True)
class WaveformValidationResult:
    rmse_uv: float
    correlation: float
    gain_error_percent: float
    offset_uv: float
    passed: bool


@dataclass(frozen=True)
class AcquisitionValidationResult:
    correlation: float
    amplitude_ratio: float
    timing_offset_ms: float
    passed: bool


@dataclass(frozen=True)
class DetectionValidationResult:
    expected_start_seconds: float
    expected_end_seconds: float
    detected_onset_seconds: float
    latency_seconds: float
    false_alarm: bool
    passed: bool


@dataclass(frozen=True)
class ValidationSuiteResult:
    simulator_output: WaveformValidationResult
    acquisition: AcquisitionValidationResult
    event_detection: DetectionValidationResult
    log_lines: tuple[str, ...]

    @property
    def passed_count(self) -> int:
        return sum(
            result.passed
            for result in (
                self.simulator_output,
                self.acquisition,
                self.event_detection,
            )
        )

    @property
    def total_count(self) -> int:
        return 3

    @property
    def overall_passed(self) -> bool:
        return self.passed_count == self.total_count


def create_mock_simulator_measurement(
    target_uv: NDArray[np.float64],
    seed: int = SIMULATOR_MEASUREMENT_SEED,
) -> NDArray[np.float64]:
    """Return a repeatable stand-in for a measured physical simulator output."""
    target = np.asarray(target_uv, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return target * 1.002 + 0.12 + rng.normal(0.0, 0.18, target.shape)


def create_mock_acquisition_capture(
    simulator_output_uv: NDArray[np.float64],
    seed: int = ACQUISITION_CAPTURE_SEED,
    timing_offset_samples: int = 1,
) -> tuple[NDArray[np.float64], int]:
    """Return a deterministic acquisition reference; this is not acquisition device software."""
    source = np.asarray(simulator_output_uv, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] <= timing_offset_samples:
        raise ValueError("Simulator output must be channel-major and longer than the timing offset")
    if timing_offset_samples < 0:
        raise ValueError("timing_offset_samples must be non-negative")

    rng = np.random.default_rng(seed)
    acquired = source * 0.998 + 0.08 + rng.normal(0.0, 0.28, source.shape)
    if timing_offset_samples:
        acquired[:, timing_offset_samples:] = acquired[:, :-timing_offset_samples]
        acquired[:, :timing_offset_samples] = acquired[:, timing_offset_samples : timing_offset_samples + 1]
    return acquired, timing_offset_samples


def validate_simulator_output(
    target_uv: NDArray[np.float64],
    measured_uv: NDArray[np.float64],
    thresholds: WaveformThresholds = DEFAULT_WAVEFORM_THRESHOLDS,
) -> WaveformValidationResult:
    target, measured = _matching_arrays(target_uv, measured_uv)
    rmse_uv, correlation, gain, offset_uv = _waveform_metrics(target, measured)
    gain_error_percent = abs(gain - 1.0) * 100.0
    passed = (
        rmse_uv <= thresholds.max_rmse_uv
        and correlation >= thresholds.min_correlation
        and gain_error_percent <= thresholds.max_gain_error_percent
        and abs(offset_uv) <= thresholds.max_abs_offset_uv
    )
    return WaveformValidationResult(
        rmse_uv=rmse_uv,
        correlation=correlation,
        gain_error_percent=gain_error_percent,
        offset_uv=offset_uv,
        passed=passed,
    )


def validate_acquisition(
    target_uv: NDArray[np.float64],
    captured_uv: NDArray[np.float64],
    sample_rate_hz: float,
    timing_offset_samples: int,
    thresholds: AcquisitionThresholds = DEFAULT_ACQUISITION_THRESHOLDS,
) -> AcquisitionValidationResult:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    target, captured = _matching_arrays(target_uv, captured_uv)
    if timing_offset_samples < 0 or timing_offset_samples >= target.shape[1]:
        raise ValueError("timing_offset_samples is outside the waveform")

    if timing_offset_samples:
        target = target[:, :-timing_offset_samples]
        captured = captured[:, timing_offset_samples:]
    target_centred = target - float(np.mean(target))
    captured_centred = captured - float(np.mean(captured))
    target_rms = float(np.sqrt(np.mean(np.square(target_centred))))
    captured_rms = float(np.sqrt(np.mean(np.square(captured_centred))))
    amplitude_ratio = captured_rms / target_rms if target_rms else 1.0
    correlation = _correlation(target, captured)
    timing_offset_ms = timing_offset_samples * 1000.0 / sample_rate_hz
    passed = (
        correlation >= thresholds.min_correlation
        and abs(amplitude_ratio - 1.0) * 100.0
        <= thresholds.max_amplitude_ratio_error_percent
        and timing_offset_ms <= thresholds.max_timing_offset_ms
    )
    return AcquisitionValidationResult(
        correlation=correlation,
        amplitude_ratio=amplitude_ratio,
        timing_offset_ms=timing_offset_ms,
        passed=passed,
    )


def validate_event_detection(
    expected: EEGAnnotation,
    detected_onset_seconds: float = MOCK_DETECTED_ONSET_SECONDS,
    false_alarm: bool = False,
    thresholds: DetectionThresholds = DEFAULT_DETECTION_THRESHOLDS,
) -> DetectionValidationResult:
    expected_end = expected.onset_seconds + expected.duration_seconds
    latency = detected_onset_seconds - expected.onset_seconds
    passed = (
        not false_alarm
        and 0.0 <= latency <= thresholds.max_latency_seconds
        and detected_onset_seconds <= expected_end
    )
    return DetectionValidationResult(
        expected_start_seconds=expected.onset_seconds,
        expected_end_seconds=expected_end,
        detected_onset_seconds=detected_onset_seconds,
        latency_seconds=latency,
        false_alarm=false_alarm,
        passed=passed,
    )


def run_validation_suite(data: EEGData) -> ValidationSuiteResult:
    """Run all three software-only checks against one known synthetic dataset."""
    expected_event = next(
        (item for item in data.annotations if "SYNTHETIC" in item.description.upper()),
        None,
    )
    if expected_event is None:
        raise ValueError("Validation requires a labelled SYNTHETIC test event")

    measured = create_mock_simulator_measurement(data.samples_uv)
    level_1 = validate_simulator_output(data.samples_uv, measured)
    captured, offset_samples = create_mock_acquisition_capture(measured)
    level_2 = validate_acquisition(
        data.samples_uv,
        captured,
        data.sample_rate_hz,
        offset_samples,
    )
    level_3 = validate_event_detection(expected_event)
    results = (level_1, level_2, level_3)
    log_lines = (
        "REFERENCE  Fixed seeds and explicit thresholds loaded",
        f"LEVEL 1    {'PASS' if level_1.passed else 'FAIL'}  Target vs mock measured output",
        f"LEVEL 2    {'PASS' if level_2.passed else 'FAIL'}  Target vs mock acquisition capture",
        f"LEVEL 3    {'PASS' if level_3.passed else 'FAIL'}  Known event vs mock detection",
        f"OVERALL    {sum(item.passed for item in results)}/3 checks passed",
        "NOTE       Software-only reference; no physical hardware or clinical detection",
    )
    return ValidationSuiteResult(level_1, level_2, level_3, log_lines)


def _matching_arrays(
    expected: NDArray[np.float64], actual: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    first = np.asarray(expected, dtype=np.float64)
    second = np.asarray(actual, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Waveforms must have matching (channels, samples) shapes")
    return first, second


def _waveform_metrics(
    expected: NDArray[np.float64], actual: NDArray[np.float64]
) -> tuple[float, float, float, float]:
    expected_flat = expected.ravel()
    actual_flat = actual.ravel()
    expected_centred = expected_flat - float(np.mean(expected_flat))
    denominator = float(np.dot(expected_centred, expected_centred))
    gain = (
        float(np.dot(expected_centred, actual_flat - float(np.mean(actual_flat))) / denominator)
        if denominator
        else 1.0
    )
    offset = float(np.mean(actual_flat) - gain * np.mean(expected_flat))
    rmse = float(np.sqrt(np.mean(np.square(actual_flat - expected_flat))))
    return rmse, _correlation(expected, actual), gain, offset


def _correlation(expected: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    expected_flat = expected.ravel()
    actual_flat = actual.ravel()
    expected_std = float(np.std(expected_flat))
    actual_std = float(np.std(actual_flat))
    if expected_std == 0.0 or actual_std == 0.0:
        return 1.0 if np.array_equal(expected_flat, actual_flat) else 0.0
    return float(np.corrcoef(expected_flat, actual_flat)[0, 1])
