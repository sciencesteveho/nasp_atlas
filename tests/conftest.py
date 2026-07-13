"""Shared pytest environment configuration."""

from __future__ import annotations

import os


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
