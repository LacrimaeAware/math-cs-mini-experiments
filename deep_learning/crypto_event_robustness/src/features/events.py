from __future__ import annotations

import numpy as np
import pandas as pd


def add_manual_seed_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def c(name):
        return out[name] if name in out.columns else pd.Series(np.nan, index=out.index)

    # Basic human-readable event flags. These are not trusted by themselves; the miner tests them.
    out["seed_dip_s7_uptrend_s60"] = ((c("ret_sum_7") < c("ret_sum_7").rolling(252, min_periods=50).quantile(0.30)) & (c("trend_slope_60") > 0)).astype(float)
    out["seed_dip_s3_uptrend_s30"] = ((c("ret_sum_3") < c("ret_sum_3").rolling(252, min_periods=50).quantile(0.25)) & (c("trend_slope_30") > 0)).astype(float)
    out["seed_panic_s7"] = ((c("ret_sum_7") < c("ret_sum_7").rolling(252, min_periods=50).quantile(0.10)) & (c("ret_std_14") > c("ret_std_14").rolling(252, min_periods=50).quantile(0.70))).astype(float)
    out["seed_compression_s7_s60"] = (c("vol_compression_7_14") < c("vol_compression_7_14").rolling(252, min_periods=50).quantile(0.25)).astype(float) if "vol_compression_7_14" in out else 0.0
    out["seed_breakout_s30"] = ((c("price_position_30") > 0.85) & (c("ret_sum_7") > 0)).astype(float)
    out["seed_near_low_s30"] = (c("price_position_30") < 0.20).astype(float)
    out["seed_near_high_s30"] = (c("price_position_30") > 0.80).astype(float)
    out["seed_chop_s30"] = ((c("ret_std_30") > c("ret_std_30").rolling(252, min_periods=50).quantile(0.40)) & (c("ret_sum_30").abs() < c("ret_std_30") * 1.5)).astype(float)
    for d in range(7):
        out[f"seed_dow_{d}"] = (out.index.dayofweek == d).astype(float)
    return out
