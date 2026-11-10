"""Fetch the hand landmark model ahead of time.

    python scripts/get_model.py

The first real run does this automatically. Use this when you want the download
out of the way, or to check the network path works before you need it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handportal.tracker import MODEL_URL, ensure_model  # noqa: E402


def main() -> int:
    print(f"source: {MODEL_URL}")
    try:
        path = ensure_model()
    except RuntimeError as exc:
        print(f"\nFailed.\n{exc}", file=sys.stderr)
        return 1
    print(f"ready:  {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
