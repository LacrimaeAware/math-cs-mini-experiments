from __future__ import annotations

import numpy as np
import pandas as pd


def toy_long_cash_equity(df: pd.DataFrame, prob: np.ndarray, active: np.ndarray, horizon: int, cost_bps: float = 10.0, initial_cash: float = 1000.0) -> np.ndarray:
    """Simple illustrative long/cash simulation on daily rows.

    For h=1 this is closest to sensible. For h>1 it uses overlapping horizon returns,
    so treat as a toy diagnostic, not a realistic execution engine.
    """
    yret = df[f"target_return_{horizon}"].to_numpy(dtype=float)
    position = ((np.asarray(prob) >= 0.5) & active).astype(float)
    cost = cost_bps / 10000.0
    equity = np.empty(len(df), dtype=float)
    val = initial_cash
    prev_pos = 0.0
    for i in range(len(df)):
        pos = position[i]
        if pos != prev_pos:
            val *= (1.0 - cost)
        val *= float(np.exp(pos * yret[i]))
        equity[i] = val
        prev_pos = pos
    return equity
