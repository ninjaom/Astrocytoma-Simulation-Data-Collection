"""Shared (time, data) helpers -- see CODE_OVERVIEW.md for why these are split out."""

from __future__ import annotations

import numpy as np


def squeeze_monitor_data(data):
    """TVB monitor output is (T, state_vars, nodes, modes); collapse to (T, nodes)."""
    return np.asarray(data)[:, 0, :, 0]


def drop_burn_in(time, data, burn_in_ms):
    """Discard the initial transient (fixed initial condition) before analysis."""
    time = np.asarray(time)
    cutoff = np.searchsorted(time, time[0] + burn_in_ms)
    return time[cutoff:], data[cutoff:]
