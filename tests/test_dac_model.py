from __future__ import annotations

import unittest

import numpy as np

from eeg_simulator.dac_model import VirtualDAC


class VirtualDACTests(unittest.TestCase):
    def test_zero_maps_to_midscale(self) -> None:
        dac = VirtualDAC()
        self.assertEqual(dac.microvolts_to_code(0.0), dac.midscale_code)

    def test_codes_are_clipped_to_valid_range(self) -> None:
        dac = VirtualDAC()
        codes = np.asarray(dac.microvolts_to_code([-1_000.0, -200.0, 0.0, 200.0, 1_000.0]))
        self.assertTrue(np.all(codes >= 0))
        self.assertTrue(np.all(codes <= dac.max_code))
        self.assertEqual(int(codes[0]), 0)
        self.assertEqual(int(codes[-1]), dac.max_code)

    def test_mapping_is_monotonic(self) -> None:
        codes = np.asarray(VirtualDAC().microvolts_to_code(np.linspace(-200.0, 200.0, 1001)))
        self.assertTrue(np.all(np.diff(codes.astype(np.int64)) >= 0))


if __name__ == "__main__":
    unittest.main()
