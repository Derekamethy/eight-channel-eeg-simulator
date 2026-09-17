from __future__ import annotations

import unittest

import numpy as np

from eeg_simulator.device import DeviceState, MockSimulatorDevice
from eeg_simulator.protocol import decode_message, encode_message


class ProtocolTests(unittest.TestCase):
    def test_encode_and_decode(self) -> None:
        encoded = encode_message("config", fs=256, amp=1.0, channels=8)
        self.assertEqual(encoded, "CONFIG FS=256 AMP=1.0 CHANNELS=8")
        decoded = decode_message(encoded)
        self.assertEqual(decoded.command, "CONFIG")
        self.assertEqual(decoded.parameters["FS"], "256")
        self.assertEqual(decoded.parameters["CHANNELS"], "8")

    def test_mock_device_state_transitions_and_log(self) -> None:
        log: list[str] = []
        device = MockSimulatorDevice(log.append)
        self.assertEqual(device.get_status().state, DeviceState.DISCONNECTED)
        device.connect()
        self.assertEqual(device.get_status().state, DeviceState.IDLE)
        device.configure(256, 1.0, ("F3", "F4"), False)
        device.upload_waveform(np.zeros((2, 256), dtype=np.float64))
        self.assertEqual(device.get_status().state, DeviceState.READY)
        device.start()
        self.assertEqual(device.get_status().state, DeviceState.RUNNING)
        device.pause()
        self.assertEqual(device.get_status().state, DeviceState.PAUSED)
        device.start()
        device.stop()
        self.assertEqual(device.get_status().state, DeviceState.READY)
        device.reset()
        self.assertEqual(device.get_status().state, DeviceState.IDLE)
        device.disconnect()
        self.assertEqual(device.get_status().state, DeviceState.DISCONNECTED)
        self.assertIn("TX > CONFIG FS=256 AMP=1 CHANNELS=2 LOOP=0", log)
        self.assertIn("RX < READY", log)

    def test_start_requires_uploaded_waveform(self) -> None:
        device = MockSimulatorDevice()
        device.connect()
        with self.assertRaises(RuntimeError):
            device.start()


if __name__ == "__main__":
    unittest.main()
