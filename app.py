"""Application entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eight-channel EEG Simulator desktop proof of concept")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a generate/connect/start validation flow and exit automatically",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Save a screenshot during --smoke-test (validation helper)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test and os.name != "nt":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    try:
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        import pyqtgraph  # noqa: F401
    except ImportError as exc:
        print(
            "Missing GUI dependency. Install with 'python -m pip install -r requirements.txt'.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1

    from eeg_simulator.gui import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("EEG Simulator Engineering Prototype")
    window = MainWindow()
    window.show()

    if args.smoke_test:
        def run_scenario() -> None:
            window.run_smoke_scenario()

        def finish() -> None:
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                if not window.grab().save(str(args.screenshot)):
                    print(f"Could not save screenshot to {args.screenshot}", file=sys.stderr)
                    app.exit(2)
                    return
            app.quit()

        QTimer.singleShot(100, run_scenario)
        QTimer.singleShot(1800, finish)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
