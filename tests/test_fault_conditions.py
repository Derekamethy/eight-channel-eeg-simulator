from __future__ import annotations

import unittest

import numpy as np

from eeg_simulator.fault_conditions import (
    ContactState,
    ElectrodeConditionModel,
    InterferenceMode,
)
from eeg_simulator.synthetic_eeg import ACTIVE_CHANNELS, generate_synthetic_eeg


class FaultConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = generate_synthetic_eeg(duration_seconds=2.0)

    def setUp(self) -> None:
        self.model = ElectrodeConditionModel(ACTIVE_CHANNELS)

    def test_normal_mode_preserves_all_samples(self) -> None:
        output = self.model.apply(self.data.samples_uv, self.data.sample_rate_hz)
        np.testing.assert_array_equal(output, self.data.samples_uv)

    def test_high_impedance_changes_only_selected_channel(self) -> None:
        self.model.set_contact_state("C3", ContactState.HIGH)
        output = self.model.apply(self.data.samples_uv, self.data.sample_rate_hz)
        self.assertFalse(np.array_equal(output[2], self.data.samples_uv[2]))
        np.testing.assert_array_equal(output[:2], self.data.samples_uv[:2])
        np.testing.assert_array_equal(output[3:], self.data.samples_uv[3:])

    def test_high_impedance_effect_is_deterministic(self) -> None:
        self.model.set_contact_state("F3", ContactState.VERY_HIGH)
        first = self.model.apply(self.data.samples_uv, self.data.sample_rate_hz)
        second = self.model.apply(self.data.samples_uv, self.data.sample_rate_hz)
        np.testing.assert_array_equal(first, second)

    def test_lead_off_replaces_only_selected_channel(self) -> None:
        self.model.set_contact_state("C4", ContactState.LEAD_OFF)
        output = self.model.apply(self.data.samples_uv, self.data.sample_rate_hz)
        self.assertLess(np.corrcoef(output[3], self.data.samples_uv[3])[0, 1], 0.5)
        np.testing.assert_array_equal(output[0], self.data.samples_uv[0])
        self.assertIn("C4: LEAD OFF", self.model.abnormal_summary)

    def test_mains_interference_is_added_not_replacement(self) -> None:
        self.model.set_interference_mode(InterferenceMode.MAINS_50_HZ)
        output = self.model.apply(self.data.samples_uv, self.data.sample_rate_hz)
        self.assertFalse(np.array_equal(output, self.data.samples_uv))
        self.assertGreater(np.corrcoef(output.ravel(), self.data.samples_uv.ravel())[0, 1], 0.8)

    def test_multi_condition_and_reset(self) -> None:
        self.model.set_contact_state("F4", ContactState.HIGH)
        self.model.set_contact_state("C3", ContactState.LEAD_OFF)
        self.model.set_interference_mode(InterferenceMode.MAINS_50_HZ)
        self.assertEqual(len(self.model.abnormal_summary), 3)
        self.model.reset()
        self.assertEqual(self.model.abnormal_summary, ())
        np.testing.assert_array_equal(
            self.model.apply(self.data.samples_uv, self.data.sample_rate_hz),
            self.data.samples_uv,
        )


if __name__ == "__main__":
    unittest.main()
