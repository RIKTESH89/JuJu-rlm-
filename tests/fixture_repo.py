"""Builds a small, known repository for the end-to-end tests to work on."""

import json
from pathlib import Path

PY_FILE_COUNT = 7
BIG_FILE_LINES = 5000
BIG_FILE_NAME = "huge.py"
TARGET_FUNCTION = "compute_checksum"
SECRET_RETURN = 8675309


def build(root: Path) -> Path:
    """Create the fixture tree and return its root."""
    root.mkdir(parents=True, exist_ok=True)

    for n in range(PY_FILE_COUNT - 1):
        (root / f"module_{n}.py").write_text(
            f"def helper_{n}(value):\n    return value + {n}\n"
        )

    # One large file: every line uniquely marked, so a test can count how much
    # of it leaked into the transcript.
    lines = [f"# MARKER_{i:05d} filler line for the context economy test"
             for i in range(BIG_FILE_LINES)]
    lines[BIG_FILE_LINES // 2] = (
        f"def {TARGET_FUNCTION}(data):\n"
        f"    # the only function that matters in this file\n"
        f"    return {SECRET_RETURN}"
    )
    (root / BIG_FILE_NAME).write_text("\n".join(lines) + "\n")

    # Invalid JSON: a trailing comma. json.loads raises on it.
    (root / "broken.json").write_text('{\n  "total": 1234,\n  "name": "widget",\n}\n')

    (root / "notes.md").write_text("Fixture repository for scrivo end-to-end tests.\n")
    return root
