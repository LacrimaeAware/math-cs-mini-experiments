from __future__ import annotations

import numpy as np
import pandas as pd


def add_targets(df: pd.DataFrame, horizons) -> pd.DataFrame:
    out = df.copy()
    log_close = out["log_close"].astype(float) if "log_close" in out else np.log(out["Close"].astype(float))
    for h in horizons:
        fut = log_close.shift(-int(h))
        ret = fut - log_close
        out[f"target_return_{h}"] = ret
        out[f"target_direction_{h}"] = (ret > 0).astype(float)
        out[f"future_close_{h}"] = out["Close"].shift(-int(h))
        # Previous-move baselines as data columns.
        out[f"prev_1d_return_scaled_to_h_{h}"] = out["log_ret_1"] * int(h)
        if f"ret_sum_{h}" in out.columns:
            out[f"prev_same_h_return_{h}"] = out[f"ret_sum_{h}"]
    return out
