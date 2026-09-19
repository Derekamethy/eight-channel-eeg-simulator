from __future__ import annotations

import unittest

import numpy as np

from eeg_simulator.dac_model import VirtualDAC
from eeg_simulator.fault_conditions import ContactState, InterferenceMode
from web_demo.console_logic import build_demo_waveform, format_time, time_to_sample_index


class WebDemoLogicTests(unittest.TestCase):
    def test_default_demo_shape_and_dac_mapping(self) -> None:
        demo = build_demo_waveform()
        self.assertEqual(demo.samples_uv.shape, (8, 60 * 256))
        self.assertEqual(demo.dac_codes.shape, demo.samples_uv.shape)
        self.assertEqual(
            demo.channel_names,
            ("F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2"),
        )
        expected = VirtualDAC().microvolts_to_code(demo.samples_uv)
        np.testing.assert_array_equal(demo.dac_codes, expected)

    def test_amplitude_scale_changes_normal_waveform(self) -> None:
        baseline = build_demo_waveform(amplitude_scale=1.0)
        doubled = build_demo_waveform(amplitude_scale=2.0)
        np.testing.assert_allclose(doubled.samples_uv, baseline.samples_uv * 2.0)

    def test_single_channel_contact_condition_is_isolated(self) -> None:
        baseline = build_demo_waveform()
        changed = build_demo_waveform(
            contact_channel="F3",
            contact_state=ContactState.HIGH,
            interference_mode=InterferenceMode.NONE,
        )
        self.assertFalse(np.array_equal(changed.samples_uv[0], baseline.samples_uv[0]))
        np.testing.assert_array_equal(changed.samples_uv[1:], baseline.samples_uv[1:])

    def test_multiple_channel_conditions_apply_independently(self) -> None:
        baseline = build_demo_waveform()
        changed = build_demo_waveform(
            contact_states={
                "F3": ContactState.HIGH,
                "T4": ContactState.LEAD_OFF,
            }
        )
        f3 = changed.channel_names.index("F3")
        t4 = changed.channel_names.index("T4")
        self.assertFalse(np.array_equal(changed.samples_uv[f3], baseline.samples_uv[f3]))
        self.assertFalse(np.array_equal(changed.samples_uv[t4], baseline.samples_uv[t4]))
        for index, channel in enumerate(changed.channel_names):
            if channel not in {"F3", "T4"}:
                np.testing.assert_array_equal(
                    changed.samples_uv[index], baseline.samples_uv[index]
                )

    def test_unknown_condition_channel_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_demo_waveform(contact_states={"BAD": ContactState.HIGH})

    def test_time_to_sample_index_clamps_bounds(self) -> None:
        self.assertEqual(time_to_sample_index(-1.0, 256.0, 100), 0)
        self.assertEqual(time_to_sample_index(0.1, 256.0, 100), 26)
        self.assertEqual(time_to_sample_index(99.0, 256.0, 100), 99)

    def test_format_time_matches_console_clock(self) -> None:
        self.assertEqual(format_time(0.0), "00:00.000")
        self.assertEqual(format_time(61.125), "01:01.125")


if __name__ == "__main__":
    unittest.main()
