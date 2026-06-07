from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float).clip(lower=EPS)
    high = out["High"].astype(float).clip(lower=EPS)
    low = out["Low"].astype(float).clip(lower=EPS)
    open_ = out["Open"].astype(float).clip(lower=EPS)
    volume = out["Volume"].astype(float).clip(lower=0)

    out["log_close"] = np.log(close)
    out["log_volume"] = np.log1p(volume)
    out["log_ret_1"] = out["log_close"].diff()
    out["open_close_ret"] = np.log(close / open_)
    out["high_low_range"] = np.log(high / low)
    out["close_to_high"] = np.log(close / high)
    out["close_to_low"] = np.log(close / low)
    out["volume_chg_1"] = out["log_volume"].diff()

    dow = out.index.dayofweek.astype(float)
    month = out.index.month.astype(float)
    doy = out.index.dayofyear.astype(float)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for d in range(7):
        out[f"dow_is_{d}"] = (out.index.dayofweek == d).astype(float)

    for c in list(out.columns):
        if c.startswith("ctx_") and c.endswith("_close"):
            s = out[c].astype(float).clip(lower=EPS)
            out[f"{c}_log"] = np.log(s)
            out[f"{c}_ma_gap_30"] = np.log(s) - np.log(s.rolling(30).mean().clip(lower=EPS))
    return out
