# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible entry point. Prefer ``uv run maskdecide``."""

from maskdecide.api import app

__all__ = ["app"]
