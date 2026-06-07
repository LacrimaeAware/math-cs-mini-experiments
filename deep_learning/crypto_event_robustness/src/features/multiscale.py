from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def rolling_slope(x: pd.Series, window: int) -> pd.Series:
    t = np.arange(window, dtype=float)
    t = t - t.mean()
    denom = float((t * t).sum())

    def _slope(values):
        if np.any(~np.isfinite(values)):
            return np.nan
        y = values - np.mean(values)
        return float(np.dot(t, y) / denom)

    return x.rolling(window).apply(_slope, raw=True)


def rolling_skew(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).skew()


def rolling_kurt(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).kurt()


def add_multiscale_features(df: pd.DataFrame, scales) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float).clip(lower=EPS)
    log_close = np.log(close)
    ret = out["log_ret_1"] if "log_ret_1" in out else log_close.diff()
    log_volume = out.get("log_volume", np.log1p(out["Volume"].astype(float)))

    new_cols = {}
    for w in scales:
        w = int(w)
        if w < 1:
            continue
        roll_ret = ret.rolling(w)
        ma = close.rolling(w).mean().clip(lower=EPS)
        hi = close.rolling(w).max().clip(lower=EPS)
        lo = close.rolling(w).min().clip(lower=EPS)
        pos = (close - lo) / (hi - lo + EPS)
        roll_vol = log_volume.rolling(w)
        new_cols[f"ret_sum_{w}"] = log_close - log_close.shift(w)
        new_cols[f"ret_mean_{w}"] = roll_ret.mean()
        new_cols[f"ret_std_{w}"] = roll_ret.std()
        if w >= 5:
            new_cols[f"ret_skew_{w}"] = rolling_skew(ret, w)
            new_cols[f"ret_kurt_{w}"] = rolling_kurt(ret, w)
        if w >= 3:
            new_cols[f"trend_slope_{w}"] = rolling_slope(log_close, w)
        else:
            new_cols[f"trend_slope_{w}"] = (log_close - log_close.shift(w)) / max(w, 1)
        new_cols[f"ma_gap_{w}"] = log_close - np.log(ma)
        new_cols[f"zscore_close_{w}"] = (log_close - log_close.rolling(w).mean()) / (log_close.rolling(w).std() + EPS)
        new_cols[f"price_position_{w}"] = pos
        new_cols[f"drawdown_from_high_{w}"] = log_close - np.log(hi)
        new_cols[f"bounce_from_low_{w}"] = log_close - np.log(lo)
        new_cols[f"hl_range_mean_{w}"] = out["high_low_range"].rolling(w).mean() if "high_low_range" in out else np.nan
        new_cols[f"volume_z_{w}"] = (log_volume - roll_vol.mean()) / (roll_vol.std() + EPS)
        new_cols[f"volume_expansion_{w}"] = out.get("volume_chg_1", log_volume.diff()).rolling(w).mean()

    out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    cross_cols = {}
    scales = [int(s) for s in scales]
    for short, long in zip(scales[:-1], scales[1:]):
        if f"ret_std_{short}" in out and f"ret_std_{long}" in out:
            cross_cols[f"vol_compression_{short}_{long}"] = out[f"ret_std_{short}"] / (out[f"ret_std_{long}"] + EPS)
        if f"trend_slope_{short}" in out and f"trend_slope_{long}" in out:
            cross_cols[f"trend_agree_{short}_{long}"] = np.sign(out[f"trend_slope_{short}"]) * np.sign(out[f"trend_slope_{long}"])
            cross_cols[f"trend_diff_{short}_{long}"] = out[f"trend_slope_{short}"] - out[f"trend_slope_{long}"]
        if f"ret_sum_{short}" in out and f"ret_sum_{long}" in out:
            cross_cols[f"ret_ratio_{short}_{long}"] = out[f"ret_sum_{short}"] / (out[f"ret_sum_{long}"].abs() + EPS)

    return pd.concat([out, pd.DataFrame(cross_cols, index=out.index)], axis=1)
