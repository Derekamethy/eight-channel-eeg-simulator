from __future__ import annotations

import unittest

from eeg_simulator.fault_conditions import ContactState
from web_demo.console_logic import build_demo_waveform
from web_demo.live_monitor import (
    MONITOR_IFRAME_HEIGHT_PX,
    build_live_monitor_html,
)


class WebLiveMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.demo = build_demo_waveform()
        self.states = {
            channel: ContactState.NORMAL for channel in self.demo.channel_names
        }

    def test_monitor_renders_waveforms_once_with_browser_animation(self) -> None:
        html = build_live_monitor_html(
            self.demo,
            self.demo.channel_names,
            self.states,
            initial_position_seconds=5.0,
            running=True,
            loop=False,
            state_label="RUNNING",
        )
        self.assertEqual(html.count("<path d="), 8)
        self.assertEqual(html.count('id="cursor"'), 1)
        self.assertIn("requestAnimationFrame(draw)", html)
        self.assertNotIn("plotly", html.lower())

    def test_monitor_reserves_footer_and_keeps_labels_outside_scaled_svg(self) -> None:
        html = build_live_monitor_html(
            self.demo,
            self.demo.channel_names,
            self.states,
            initial_position_seconds=0.0,
            running=False,
            loop=False,
            state_label="READY",
        )
        self.assertEqual(MONITOR_IFRAME_HEIGHT_PX, 468)
        self.assertNotIn("height:auto", html)
        self.assertIn("grid-template-rows:minmax(0, 1fr) 32px", html)
        svg = html.split("<svg", 1)[1].split("</svg>", 1)[0]
        self.assertNotIn("<text", svg)
        self.assertEqual(html.count('class="channel-label"'), 8)
        self.assertEqual(html.count('class="range-label"'), 8)
        self.assertEqual(html.count('class="tick-label"'), 7)
        self.assertIn("overlay-title", html)
        self.assertNotIn('<div class="top">', html)

    def test_loop_and_initial_position_are_encoded_for_front_end(self) -> None:
        html = build_live_monitor_html(
            self.demo,
            self.demo.channel_names,
            self.states,
            initial_position_seconds=12.5,
            running=True,
            loop=True,
            state_label="RUNNING",
        )
        self.assertIn("const loop = true;", html)
        self.assertIn("let running = true;", html)
        self.assertIn("const initial = 12.500000000;", html)

    def test_inactive_and_lead_off_channels_keep_fixed_lanes(self) -> None:
        states = dict(self.states)
        states["F3"] = ContactState.LEAD_OFF
        active = tuple(channel for channel in self.demo.channel_names if channel != "O2")
        html = build_live_monitor_html(
            self.demo,
            active,
            states,
            initial_position_seconds=0.0,
            running=False,
            loop=False,
            state_label="READY",
        )
        self.assertIn('stroke-dasharray="5 4"', html)
        self.assertIn('opacity="0.16"', html)
        for channel in self.demo.channel_names:
            self.assertIn(f">{channel}</span>", html)


if __name__ == "__main__":
    unittest.main()
