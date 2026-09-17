from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "pyproject.toml",
    "app.py",
    "eeg_simulator/__init__.py",
    "eeg_simulator/gui.py",
    "tests/test_validation.py",
    "data/synthetic_eeg_demo.npz",
    "assets/eeg_simulator_demo.png",
    ".github/workflows/ci.yml",
]
FORBIDDEN_TEXT = (
    "Neuro" + "Bell",
    "interview " + "demo",
    "interview " + "discussion",
)
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".txt"}


def fail(message: str) -> None:
    raise SystemExit(f"RELEASE CHECK FAILED: {message}")


for relative in REQUIRED:
    if not (ROOT / relative).exists():
        fail(f"missing required file: {relative}")

for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts or ".venv" in path.parts:
        continue
    if path.suffix.lower() not in TEXT_SUFFIXES:
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    for term in FORBIDDEN_TEXT:
        if term.lower() in text.lower():
            fail(f"legacy/internal wording {term!r} found in {path.relative_to(ROOT)}")

archive = np.load(ROOT / "data" / "synthetic_eeg_demo.npz", allow_pickle=False)
try:
    samples = np.asarray(archive["samples_uv"], dtype=np.float64)
    metadata_text = str(archive["metadata_json"].item())
finally:
    archive.close()

if samples.shape != (8, 60 * 256):
    fail(f"unexpected demo data shape: {samples.shape}")
if '"synthetic": true' not in metadata_text.lower():
    fail("demo dataset is not explicitly marked synthetic")

ignored = subprocess.run(
    ["git", "check-ignore", ".venv"], cwd=ROOT,
    capture_output=True, text=True, check=False,
)
if ignored.returncode != 0:
    fail(".venv is not ignored by Git")

tracked = subprocess.run(
    ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.splitlines()
if any("__pycache__" in item or item.endswith(".pyc") for item in tracked):
    fail("Python cache files are tracked")

print("Release checks passed")
