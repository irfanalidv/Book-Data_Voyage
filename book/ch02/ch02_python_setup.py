#!/usr/bin/env python3
"""Chapter 2: project scaffold demo - config, paths, and structured logging."""

from __future__ import annotations

import logging
import sys

from talentlens import DATA_DIR, REPO_ROOT, __version__, get_settings
from talentlens.paths import BOOK_DIR

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    """Configure process-wide logging once (chapters import this pattern)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    """Load settings, print resolved paths, demonstrate logging vs print."""
    try:
        settings = get_settings()
    except Exception as exc:  # noqa: BLE001 - teaching script surfaces config errors
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    configure_logging(settings.log_level)

    logger.debug("Debug line — visible when LOG_LEVEL=DEBUG")
    logger.info("Info line — this is how chapters log diagnostics")

    print("=" * 60)
    print("Data Voyage — Chapter 2 project scaffold")
    print("=" * 60)
    print(f"talentlens version: {__version__}")
    print(f"REPO_ROOT:          {REPO_ROOT}")
    print(f"DATA_DIR:           {DATA_DIR}")
    print(f"BOOK_DIR:           {BOOK_DIR}")
    print("Settings (secrets masked):")
    for key, value in settings.mask_secrets().items():
        print(f"  {key}: {value}")
    print()
    print("Project scaffold is ready. Continue with Chapter 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
