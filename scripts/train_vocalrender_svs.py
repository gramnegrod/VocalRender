#!/usr/bin/env python3
"""Thin CLI wrapper for the SVS training runner."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vocalrender.training.config import parse_script_config
from vocalrender.training.runners.svs import run


def main(argv: list[str] | None = None) -> int:
    config = parse_script_config(kind="svs", argv=argv)
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
