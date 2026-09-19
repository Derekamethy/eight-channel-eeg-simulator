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
