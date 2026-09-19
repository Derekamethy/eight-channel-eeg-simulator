from __future__ import annotations

import unittest

from eeg_simulator.fault_conditions import ContactState
from web_demo.console_logic import build_demo_waveform
from web_demo.visualization import build_monitor_figure, nice_axis_limit


class WebVisualizationTests(unittest.TestCase):
    def test_monitor_has_fixed_eight_channel_lanes(self) -> None:
        demo = build_demo_waveform()
        states = {channel: ContactState.NORMAL for channel in demo.channel_names}
        figure = build_monitor_figure(
            demo,
            demo.channel_names,
            states,
            cursor_seconds=5.0,
        )
        self.assertEqual(len(figure.data), 8)
        self.assertEqual(sum(trace.visible is not False for trace in figure.data), 8)
        self.assertEqual(figure.layout.height, 610)

    def test_inactive_channel_keeps_lane_but_hides_trace(self) -> None:
        demo = build_demo_waveform()
        states = {channel: ContactState.NORMAL for channel in demo.channel_names}
        figure = build_monitor_figure(
            demo,
            tuple(channel for channel in demo.channel_names if channel != "O2"),
            states,
            cursor_seconds=5.0,
        )
        self.assertFalse(figure.data[-1].visible)

    def test_axis_limit_is_symmetric_friendly_value(self) -> None:
        self.assertEqual(nice_axis_limit(demo := build_demo_waveform().samples_uv[0]) > 0, True)


if __name__ == "__main__":
    unittest.main()
