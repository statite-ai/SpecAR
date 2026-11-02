# SPDX-License-Identifier: Apache-2.0
"""SpecAR: distributed target blocks for fast autoregressive video generation."""
from .schedule import max_draft_run, uniform_k

__version__ = "0.1.0"
__all__ = ["uniform_k", "max_draft_run"]
