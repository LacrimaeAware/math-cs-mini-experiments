from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _fft_stats(values: np.ndarray):
    if np.any(~np.isfinite(values)):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    x = values - np.mean(values)
    if np.std(x) < EPS:
        return (np.nan, 0.0, 0.0, 0.0, 0.0, 0.0)
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    if len(power) <= 2 or power[1:].sum() <= EPS:
        return (np.nan, 0.0, 0.0, 0.0, 0.0, 0.0)
    power_no_dc = power[1:]
    idx = int(np.argmax(power_no_dc)) + 1
    freq = idx / len(x)
    period = 1.0 / max(freq, EPS)
    strength = power[idx] / (power_no_dc.sum() + EPS)
    p = power_no_dc / (power_no_dc.sum() + EPS)
    entropy = -np.sum(p * np.log(p + EPS)) / np.log(len(p) + EPS)
    phase = np.angle(fft[idx])
    # Rough energy bands: low/mid/high among non-DC bins.
    m = len(power_no_dc)
    low = power_no_dc[: max(1, m // 3)].sum() / (power_no_dc.sum() + EPS)
    mid = power_no_dc[max(1, m // 3): max(2, 2 * m // 3)].sum() / (power_no_dc.sum() + EPS)
    high = power_no_dc[max(2, 2 * m // 3):].sum() / (power_no_dc.sum() + EPS)
    return (period, strength, entropy, np.sin(phase), np.cos(phase), low - high + 0.25 * mid)


def add_fft_features(df: pd.DataFrame, windows) -> pd.DataFrame:
    out = df.copy()
    ret = out["log_ret_1"].astype(float)
    frames = []
    for w in windows:
        cols = [
            f"fft_period_{w}", f"fft_strength_{w}", f"fft_entropy_{w}",
            f"fft_phase_sin_{w}", f"fft_phase_cos_{w}", f"fft_lowhigh_balance_{w}",
        ]
        arr = np.full((len(out), len(cols)), np.nan)
        vals = ret.to_numpy(dtype=float)
        for i in range(w - 1, len(vals)):
            arr[i, :] = _fft_stats(vals[i - w + 1:i + 1])
        frames.append(pd.DataFrame(arr, index=out.index, columns=cols))
    return pd.concat([out] + frames, axis=1)


def add_multiresolution_components(df: pd.DataFrame, scales) -> pd.DataFrame:
    out = df.copy()
    log_close = out["log_close"].astype(float) if "log_close" in out else np.log(out["Close"].astype(float))
    frames = []
    cols = {}
    # Band-pass-like components via differences of moving averages.
    sorted_scales = sorted([int(s) for s in scales if int(s) >= 2])
    for s in sorted_scales:
        ma_s = log_close.rolling(s).mean()
        cols[f"band_resid_{s}"] = log_close - ma_s
    for a, b in zip(sorted_scales[:-1], sorted_scales[1:]):
        ma_a = log_close.rolling(a).mean()
        ma_b = log_close.rolling(b).mean()
        band = ma_a - ma_b
        cols[f"band_{a}_{b}"] = band
        cols[f"band_slope_{a}_{b}"] = band.diff()
        cols[f"band_energy_{a}_{b}"] = band.rolling(b).std()
    frames.append(pd.DataFrame(cols, index=out.index))
    return pd.concat([out] + frames, axis=1)
