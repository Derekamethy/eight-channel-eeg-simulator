from __future__ import annotations

import unittest

import numpy as np

from eeg_simulator.synthetic_eeg import generate_synthetic_eeg
from eeg_simulator.validation import (
    WaveformThresholds,
    create_mock_acquisition_capture,
    create_mock_simulator_measurement,
    run_validation_suite,
    validate_event_detection,
    validate_acquisition,
    validate_simulator_output,
)


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = generate_synthetic_eeg()

    def test_identical_waveforms_have_ideal_metrics(self) -> None:
        result = validate_simulator_output(self.data.samples_uv, self.data.samples_uv)
        self.assertEqual(result.rmse_uv, 0.0)
        self.assertAlmostEqual(result.correlation, 1.0)
        self.assertAlmostEqual(result.gain_error_percent, 0.0)
        self.assertAlmostEqual(result.offset_uv, 0.0)
        self.assertTrue(result.passed)

    def test_mock_measurements_are_deterministic(self) -> None:
        first = create_mock_simulator_measurement(self.data.samples_uv)
        second = create_mock_simulator_measurement(self.data.samples_uv)
        np.testing.assert_array_equal(first, second)

    def test_level_one_passes_reference_thresholds(self) -> None:
        measured = create_mock_simulator_measurement(self.data.samples_uv)
        self.assertTrue(validate_simulator_output(self.data.samples_uv, measured).passed)

    def test_acquisition_passes_amplitude_and_timing_checks(self) -> None:
        measured = create_mock_simulator_measurement(self.data.samples_uv)
        capture, offset = create_mock_acquisition_capture(measured)
        result = validate_acquisition(
            self.data.samples_uv, capture, self.data.sample_rate_hz, offset
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.timing_offset_ms, 1000.0 / 256.0)
        self.assertLess(abs(result.amplitude_ratio - 1.0), 0.03)

    def test_detection_latency_matches_known_event(self) -> None:
        result = validate_event_detection(self.data.annotations[0])
        self.assertTrue(result.passed)
        self.assertEqual(result.latency_seconds, 0.75)
        self.assertFalse(result.false_alarm)

    def test_overall_suite_passes_three_of_three(self) -> None:
        result = run_validation_suite(self.data)
        self.assertTrue(result.overall_passed)
        self.assertEqual(result.passed_count, 3)

    def test_deliberately_strict_threshold_fails(self) -> None:
        measured = create_mock_simulator_measurement(self.data.samples_uv)
        thresholds = WaveformThresholds(max_rmse_uv=0.01)
        self.assertFalse(
            validate_simulator_output(self.data.samples_uv, measured, thresholds).passed
        )


if __name__ == "__main__":
    unittest.main()
