from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class WebDemoAppTests(unittest.TestCase):
    def make_app(self) -> AppTest:
        return AppTest.from_file(
            ROOT / "web_demo" / "app.py",
            default_timeout=20,
        ).run()

    def test_default_console_renders_without_exception(self) -> None:
        app = self.make_app()
        self.assertEqual(list(app.exception), [])
        self.assertIsNotNone(app.get_by_key("connect_btn"))
        self.assertIsNotNone(app.get_by_key("start_btn"))
        self.assertIsNotNone(app.get_by_key("validation_btn"))

    def test_compact_controls_preserve_channel_editing_and_guard(self) -> None:
        from eeg_simulator.fault_conditions import ContactState
        from eeg_simulator.synthetic_eeg import ACTIVE_CHANNELS

        app = self.make_app()
        for channel in ACTIVE_CHANNELS:
            app.get_by_key(f"condition_{channel}").select(ContactState.VERY_HIGH)
        app.run()
        for channel in ACTIVE_CHANNELS:
            self.assertEqual(app.session_state[f"condition_{channel}"], ContactState.VERY_HIGH)
            app.get_by_key(f"active_{channel}").uncheck().run()
        self.assertTrue(app.session_state["active_O2"])
        app.get_by_key("reset_conditions_btn").click().run()
        for channel in ACTIVE_CHANNELS:
            self.assertEqual(app.session_state[f"condition_{channel}"], ContactState.NORMAL)
        self.assertEqual(list(app.exception), [])

    def test_secondary_content_is_in_popovers_and_playback_has_help(self) -> None:
        app = self.make_app()
        self.assertEqual([p.proto.popover.label for p in app.get("popover")],
                         ["Protocol log", "Validation"])
        for key, help_text in (("start_btn", "Start"), ("pause_btn", "Pause"),
                               ("stop_btn", "Stop"), ("reset_btn", "Reset")):
            self.assertEqual(app.get_by_key(key).help, help_text)

    def test_signal_chain_opens_readable_sequential_dialog(self) -> None:
        app = self.make_app()
        app.get_by_key("signal_chain_btn").click().run()
        self.assertEqual(list(app.exception), [])
        content = "".join(element.value for element in app.get("html"))
        self.assertIn('<ol class="chain"', content)
        self.assertEqual(content.count('class="chain-step"'), 7)
        self.assertLess(content.index("EEG source"), content.index("EEG acquisition"))

    def test_connect_then_start_runs_without_exception(self) -> None:
        app = self.make_app()
        app.get_by_key("connect_btn").click().run(timeout=20)
        self.assertEqual(list(app.exception), [])
        app.get_by_key("start_btn").click().run(timeout=20)
        self.assertEqual(list(app.exception), [])

    def test_validation_button_runs_reference_suite(self) -> None:
        app = self.make_app()
        app.get_by_key("validation_btn").click().run(timeout=20)
        self.assertEqual(list(app.exception), [])
        self.assertIsNotNone(app.session_state["validation_result"])


if __name__ == "__main__":
    unittest.main()
