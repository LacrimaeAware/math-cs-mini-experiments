from __future__ import annotations

import numpy as np
import pandas as pd

from .basic import add_basic_features
from .multiscale import add_multiscale_features
from .frequency import add_fft_features, add_multiresolution_components
from .events import add_manual_seed_events
from .learned import fit_transform_learned_events
from ..targets import add_targets


def build_feature_frame(raw: pd.DataFrame, config) -> pd.DataFrame:
    df = raw.copy()
    df = add_basic_features(df)
    df = add_multiscale_features(df, config.scales)
    df = add_multiresolution_components(df, config.scales)
    df = add_fft_features(df, config.fft_windows)
    if config.include_manual_seed_events:
        df = add_manual_seed_events(df)
    df = add_targets(df, config.horizons)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def split_chronological(df: pd.DataFrame, config):
    target_cols = [f"target_return_{h}" for h in config.horizons] + [f"target_direction_{h}" for h in config.horizons]
    keep = df.dropna(subset=target_cols).copy()
    feature_cols = infer_feature_columns(keep)
    required = [c for c in feature_cols if c.startswith(("ret_sum_", "ret_std_", "trend_slope_"))][:5]
    if required:
        keep = keep.dropna(subset=required)
    n = len(keep)
    n_train = int(n * config.train_frac)
    n_val = int(n * config.val_frac)
    train = keep.iloc[:n_train].copy()
    val = keep.iloc[n_train:n_train+n_val].copy()
    test = keep.iloc[n_train+n_val:].copy()
    # Learned events fit on train only.
    train, val, test = fit_transform_learned_events(train, val, test, config)
    return train, val, test


def infer_feature_columns(df: pd.DataFrame):
    exclude_prefixes = ("target_", "future_close_")
    exclude = {"Open", "High", "Low", "Close", "Volume"}
    cols = []
    for c in df.columns:
        if c in exclude or any(c.startswith(p) for p in exclude_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().sum() > 0:
                cols.append(c)
    return cols


def feature_groups(feature_cols):
    groups = {
        "trend_momentum": [], "dip_bounce": [], "volatility": [], "frequency": [],
        "manual_events": [], "learned_events": [], "context": [], "basic": [],
    }
    for c in feature_cols:
        if c.startswith("ctx_"):
            groups["context"].append(c)
        elif c.startswith("seed_"):
            groups["manual_events"].append(c)
        elif c.startswith("learned_"):
            groups["learned_events"].append(c)
        elif c.startswith("fft_") or c.startswith("band_"):
            groups["frequency"].append(c)
        elif any(k in c for k in ["drawdown", "bounce", "price_position"]):
            groups["dip_bounce"].append(c)
        elif any(k in c for k in ["vol", "std", "compression", "hl_range", "volume"]):
            groups["volatility"].append(c)
        elif any(k in c for k in ["ret_sum", "ret_mean", "trend", "ma_gap", "zscore"]):
            groups["trend_momentum"].append(c)
        else:
            groups["basic"].append(c)
    return groups
