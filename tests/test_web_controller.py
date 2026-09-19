from __future__ import annotations

import unittest

import numpy as np

from eeg_simulator.device.base import DeviceState
from eeg_simulator.playback import PlaybackState
from web_demo.controller import WebSimulatorController


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class WebSimulatorControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.controller = WebSimulatorController(clock=self.clock)
        self.waveform = np.zeros((2, 100), dtype=np.float64)
        self.signature = ("demo", 1)

    def test_connect_start_pause_resume_stop_flow(self) -> None:
        self.controller.connect()
        self.assertEqual(self.controller.status.state, DeviceState.IDLE)

        self.controller.start(
            self.waveform,
            sample_rate_hz=10.0,
            amplitude_scale=1.0,
            channels=("F3", "F4"),
            loop=False,
            signature=self.signature,
        )
        self.assertEqual(self.controller.status.state, DeviceState.RUNNING)
        self.clock.advance(0.5)
        self.assertEqual(self.controller.snapshot().sample_index, 5)

        self.controller.pause()
        self.assertEqual(self.controller.status.state, DeviceState.PAUSED)
        paused_index = self.controller.snapshot().sample_index
        self.clock.advance(1.0)
        self.assertEqual(self.controller.snapshot().sample_index, paused_index)

        self.controller.start(
            self.waveform,
            sample_rate_hz=10.0,
            amplitude_scale=1.0,
            channels=("F3", "F4"),
            loop=False,
            signature=self.signature,
        )
        self.clock.advance(0.5)
        self.assertGreater(self.controller.snapshot().sample_index, paused_index)

        self.controller.stop()
        self.assertEqual(self.controller.status.state, DeviceState.READY)
        self.assertEqual(self.controller.playback.state, PlaybackState.STOPPED)
        self.assertEqual(self.controller.snapshot().sample_index, 0)

    def test_start_requires_connection(self) -> None:
        with self.assertRaises(RuntimeError):
            self.controller.start(
                self.waveform,
                sample_rate_hz=10.0,
                amplitude_scale=1.0,
                channels=("F3", "F4"),
                loop=False,
                signature=self.signature,
            )

    def test_changed_signature_reconfigures_after_pause(self) -> None:
        self.controller.connect()
        self.controller.start(
            self.waveform,
            10.0,
            1.0,
            ("F3", "F4"),
            False,
            self.signature,
        )
        self.controller.pause()
        self.controller.start(
            self.waveform * 2.0,
            10.0,
            2.0,
            ("F3", "F4"),
            False,
            ("demo", 2),
        )
        self.assertEqual(self.controller.status.state, DeviceState.RUNNING)
        self.assertEqual(self.controller.prepared_signature, ("demo", 2))

    def test_disconnect_stops_playback_and_clears_prepared_signature(self) -> None:
        self.controller.connect()
        self.controller.start(
            self.waveform,
            10.0,
            1.0,
            ("F3", "F4"),
            True,
            self.signature,
        )
        self.controller.disconnect()
        self.assertEqual(self.controller.status.state, DeviceState.DISCONNECTED)
        self.assertEqual(self.controller.playback.state, PlaybackState.STOPPED)
        self.assertIsNone(self.controller.prepared_signature)

    def test_protocol_log_is_persistent_and_clearable(self) -> None:
        self.controller.connect()
        self.assertTrue(any("CONNECT" in line for line in self.controller.logs))
        self.controller.clear_logs()
        self.assertEqual(self.controller.logs, [])


if __name__ == "__main__":
    unittest.main()
