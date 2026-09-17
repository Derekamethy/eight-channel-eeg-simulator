from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from eeg_simulator.eeg_data import EEGData
from eeg_simulator.synthetic_eeg import ACTIVE_CHANNELS, generate_synthetic_eeg


class SyntheticEEGTests(unittest.TestCase):
    def test_default_shape_channels_and_metadata(self) -> None:
        data = generate_synthetic_eeg()
        self.assertEqual(data.channel_names, ACTIVE_CHANNELS)
        self.assertEqual(data.samples_uv.shape, (8, 60 * 256))
        self.assertEqual(data.sample_rate_hz, 256)
        self.assertEqual(data.reference, "Cz")
        self.assertEqual(data.ground, "Pz")
        self.assertTrue(data.metadata["synthetic"])
        self.assertIn("SYNTHETIC DEMO EVENT", data.annotations[0].description)

    def test_generation_is_deterministic_and_event_is_visible(self) -> None:
        first = generate_synthetic_eeg()
        second = generate_synthetic_eeg()
        np.testing.assert_array_equal(first.samples_uv, second.samples_uv)
        event = first.samples_uv[:, 20 * 256 : 30 * 256]
        background = first.samples_uv[:, 10 * 256 : 20 * 256]
        self.assertGreater(float(np.std(event)), float(np.std(background)) * 1.25)

    def test_amplitude_scaling(self) -> None:
        data = generate_synthetic_eeg(duration_seconds=1.0)
        scaled = data.scaled(0.5)
        np.testing.assert_allclose(scaled.samples_uv, data.samples_uv * 0.5)
        np.testing.assert_array_equal(data.samples_uv, generate_synthetic_eeg(duration_seconds=1.0).samples_uv)

    def test_npz_round_trip(self) -> None:
        data = generate_synthetic_eeg(duration_seconds=1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = data.save_npz(Path(directory) / "demo.npz")
            loaded = EEGData.load_npz(path)
        self.assertEqual(loaded.channel_names, data.channel_names)
        self.assertEqual(loaded.reference, "Cz")
        np.testing.assert_allclose(loaded.samples_uv, data.samples_uv)


if __name__ == "__main__":
    unittest.main()
