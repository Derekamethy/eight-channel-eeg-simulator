from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class WebDemoAppTests(unittest.TestCase):
    def test_default_page_renders_without_exception(self) -> None:
        app = AppTest.from_file(ROOT / "web_demo" / "app.py", default_timeout=15).run()
        self.assertEqual(list(app.exception), [])

    def test_mock_device_button_runs_without_exception(self) -> None:
        app = AppTest.from_file(ROOT / "web_demo" / "app.py", default_timeout=15).run()
        app.button[0].click().run(timeout=15)
        self.assertEqual(list(app.exception), [])


if __name__ == "__main__":
    unittest.main()
