"""
Hierarchical Crypto Market Structure Lab

This project implements a stacked market-structure forecasting pipeline:
raw data -> multi-scale statistics -> regimes/hypercategories -> formula/interactions
-> specialist experts -> gated final prediction.

The point is not to promise a trading edge. It is to test whether conditional,
multi-scale structure beats simple persistence baselines under chronological validation.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from numpy.lib.stride_tricks import sliding_window_view
except Exception:  # pragma: no cover
    sliding_window_view = None

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class LabConfig:
    mode: str = "medium"
    ticker: str = "BTC-USD"
    start: str = "2016-01-01"
    end: Optional[str] = None
    output_root: str = "hierarchical_market_outputs"
    seed: int = 42
    cost_bps: float = 10.0
    use_synthetic: bool = False
    local_csv: Optional[str] = None

    # stable default context assets available from yfinance
    context_tickers: Tuple[str, ...] = (
        "ETH-USD",
        "SPY",
        "QQQ",
        "GLD",
        "TLT",
        "DX-Y.NYB",
        "^VIX",
    )


@dataclass
class ModeSettings:
    horizons: Tuple[int, ...]
    scales: Tuple[int, ...]
    spectral_windows: Tuple[int, ...]
    n_random_shape_kernels: int
    n_formula_base_features: int
    n_formula_category_features: int
    max_selected_formulas: int
    n_latent_regimes: int
    min_specialist_train: int
    min_event_val: int
    min_active_coverage: float
    include_neural: bool
    neural_epochs: int
    neural_patience: int
    include_walk_forward: bool
    include_ablation: bool


def get_settings(mode: str) -> ModeSettings:
    mode = mode.lower().strip()
    if mode == "quick":
        return ModeSettings(
            horizons=(3,),
            scales=(1, 2, 3, 7, 14, 30, 60),
            spectral_windows=(14, 30),
            n_random_shape_kernels=6,
            n_formula_base_features=18,
            n_formula_category_features=10,
            max_selected_formulas=35,
            n_latent_regimes=4,
            min_specialist_train=120,
            min_event_val=35,
            min_active_coverage=0.08,
            include_neural=False,
            neural_epochs=30,
            neural_patience=6,
            include_walk_forward=False,
            include_ablation=False,
        )
    if mode == "medium":
        return ModeSettings(
            horizons=(1, 2, 3, 7, 14),
            scales=(1, 2, 3, 7, 14, 30, 60, 120, 240),
            spectral_windows=(14, 30, 60, 120, 240),
            n_random_shape_kernels=64,
            n_formula_base_features=55,
            n_formula_category_features=30,
            max_selected_formulas=120,
            n_latent_regimes=6,
            min_specialist_train=160,
            min_event_val=40,
            min_active_coverage=0.08,
            include_neural=True,
            neural_epochs=50,
            neural_patience=8,
            include_walk_forward=True,
            include_ablation=True,
        )
    if mode == "heavy":
        return ModeSettings(
            horizons=(1, 2, 3, 5, 7, 14),
            scales=(1, 2, 3, 5, 7, 14, 30, 60, 120, 240, 365),
            spectral_windows=(14, 30, 60, 120, 240, 365),
            n_random_shape_kernels=128,
            n_formula_base_features=110,
            n_formula_category_features=60,
            max_selected_formulas=260,
            n_latent_regimes=8,
            min_specialist_train=180,
            min_event_val=45,
            min_active_coverage=0.06,
            include_neural=True,
            neural_epochs=80,
            neural_patience=10,
            include_walk_forward=True,
            include_ablation=True,
        )
    raise ValueError(f"Unknown mode: {mode!r}. Use quick, medium, or heavy.")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass


def flatten_yfinance_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return OHLCV columns from yfinance, robust to multi-index columns."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return either (field, ticker) or (ticker, field)
        levels = [list(map(str, df.columns.get_level_values(i).unique())) for i in range(df.columns.nlevels)]
        if ticker in levels[0]:
            df = df[ticker]
        elif ticker in levels[-1]:
            df = df.xs(ticker, axis=1, level=-1)
        else:
            # fallback: if only one ticker is present, drop ticker level
            if df.columns.nlevels == 2:
                possible_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
                if set(map(str, df.columns.get_level_values(0))).intersection(possible_fields):
                    df = df.droplevel(1, axis=1)
                else:
                    df = df.droplevel(0, axis=1)
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Downloaded data is missing {col!r}. Columns: {df.columns.tolist()}")
    df = df[needed]
    df = df.reset_index()
    if "Date" not in df.columns:
        # yfinance index name can be Datetime or index
        df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df.sort_values("Date").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def generate_synthetic_crypto_like(n: int = 700, seed: int = 42) -> pd.DataFrame:
    """Synthetic series for smoke testing without internet."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n, freq="D")
    regimes = np.zeros(n)
    state = 0
    for i in range(n):
        if rng.random() < 0.015:
            state = rng.integers(0, 4)
        regimes[i] = state
    drift = np.choose(regimes.astype(int), [0.0008, -0.0005, 0.0000, 0.0015])
    vol = np.choose(regimes.astype(int), [0.018, 0.028, 0.011, 0.045])
    cycle = 0.006 * np.sin(np.arange(n) / 7.0) + 0.012 * np.sin(np.arange(n) / 37.0)
    ret = drift + cycle + rng.normal(0, vol)
    logp = np.log(400.0) + np.cumsum(ret)
    close = np.exp(logp)
    high = close * np.exp(np.abs(rng.normal(0.006, 0.006, n)))
    low = close / np.exp(np.abs(rng.normal(0.006, 0.006, n)))
    open_ = close / np.exp(ret) * np.exp(rng.normal(0, 0.005, n))
    volume = np.exp(16 + rng.normal(0, 0.5, n) + 8 * vol)
    return pd.DataFrame({"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})


def load_main_data(config: LabConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if config.use_synthetic:
        main = generate_synthetic_crypto_like(seed=config.seed)
        context = {}
        return main, context

    if config.local_csv:
        main = pd.read_csv(config.local_csv)
        if "Date" not in main.columns:
            raise ValueError("Local CSV must have a Date column.")
        main["Date"] = pd.to_datetime(main["Date"]).dt.tz_localize(None)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        for c in needed:
            if c not in main.columns:
                raise ValueError(f"Local CSV missing {c!r}")
        main = main[["Date"] + needed].sort_values("Date").reset_index(drop=True)
        return main, {}

    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install yfinance or use local_csv/synthetic mode.") from exc

    print(f"Main ticker: {config.ticker} | start={config.start} | end={config.end}")
    raw = yf.download(config.ticker, start=config.start, end=config.end, interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"No rows downloaded for {config.ticker}")
    main = flatten_yfinance_frame(raw, config.ticker)

    context: Dict[str, pd.DataFrame] = {}
    for tick in config.context_tickers:
        try:
            print(f"Downloading context ticker: {tick}")
            ctx_raw = yf.download(tick, start=config.start, end=config.end, interval="1d", auto_adjust=False, progress=False)
            if ctx_raw.empty:
                continue
            context[tick] = flatten_yfinance_frame(ctx_raw, tick)
        except Exception as exc:
            print(f"  skipped {tick}: {exc}")
    return main, context


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------


def rolling_slope(y: pd.Series, window: int) -> pd.Series:
    if window < 2:
        return pd.Series(np.nan, index=y.index)
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = np.sum(x * x)

    def _slope(arr: np.ndarray) -> float:
        if np.any(~np.isfinite(arr)):
            return np.nan
        yy = arr - np.mean(arr)
        return float(np.sum(x * yy) / denom)

    return y.rolling(window, min_periods=window).apply(_slope, raw=True)


def rolling_r2(y: pd.Series, window: int) -> pd.Series:
    if window < 3:
        return pd.Series(np.nan, index=y.index)
    x = np.arange(window, dtype=float)
    x = x - x.mean()

    def _r2(arr: np.ndarray) -> float:
        if np.any(~np.isfinite(arr)):
            return np.nan
        yy = arr - np.mean(arr)
        sx = np.sqrt(np.sum(x * x))
        sy = np.sqrt(np.sum(yy * yy))
        if sy == 0 or sx == 0:
            return 0.0
        corr = float(np.sum(x * yy) / (sx * sy))
        return corr * corr

    return y.rolling(window, min_periods=window).apply(_r2, raw=True)


def rolling_arg_age(y: pd.Series, window: int, mode: str) -> pd.Series:
    arr = y.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    if window <= 1:
        return pd.Series(out, index=y.index)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1 : i + 1]
        if np.any(~np.isfinite(w)):
            continue
        idx = int(np.argmax(w) if mode == "max" else np.argmin(w))
        out[i] = window - 1 - idx
    return pd.Series(out, index=y.index)


def add_basic_context_features(main: pd.DataFrame, context: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    data = main.copy()
    data = data.sort_values("Date").reset_index(drop=True)
    eps = 1e-12
    data["log_close"] = np.log(data["Close"].clip(lower=eps))
    data["log_open"] = np.log(data["Open"].clip(lower=eps))
    data["log_high"] = np.log(data["High"].clip(lower=eps))
    data["log_low"] = np.log(data["Low"].clip(lower=eps))
    data["log_volume"] = np.log1p(data["Volume"].clip(lower=0))
    data["ret_1"] = data["log_close"].diff()
    data["open_to_close"] = data["log_close"] - data["log_open"]
    data["high_low_range"] = data["log_high"] - data["log_low"]
    data["close_to_high"] = data["log_close"] - data["log_high"]
    data["close_to_low"] = data["log_close"] - data["log_low"]
    data["volume_chg"] = data["log_volume"].diff()

    # calendar cycles, used only as weak context; crypto trades every day.
    dow = data["Date"].dt.dayofweek.astype(float)
    doy = data["Date"].dt.dayofyear.astype(float)
    data["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    data["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # context asset log returns and rolling relative strength.
    for tick, ctx in context.items():
        nm = safe_name(tick).lower()
        c = ctx[["Date", "Close", "Volume"]].copy()
        c = c.sort_values("Date")
        c[f"ctx_{nm}_log_close"] = np.log(c["Close"].clip(lower=eps))
        c[f"ctx_{nm}_ret_1"] = c[f"ctx_{nm}_log_close"].diff()
        c[f"ctx_{nm}_log_volume"] = np.log1p(c["Volume"].clip(lower=0))
        c = c[["Date", f"ctx_{nm}_ret_1", f"ctx_{nm}_log_volume"]]
        data = data.merge(c, on="Date", how="left")
        data[f"ctx_{nm}_ret_1"] = data[f"ctx_{nm}_ret_1"].ffill()
        data[f"ctx_{nm}_log_volume"] = data[f"ctx_{nm}_log_volume"].ffill()
        for w in (7, 30, 90):
            data[f"rel_strength_{nm}_{w}"] = data["ret_1"].rolling(w).sum() - data[f"ctx_{nm}_ret_1"].rolling(w).sum()
            data[f"corr_{nm}_{w}"] = data["ret_1"].rolling(w).corr(data[f"ctx_{nm}_ret_1"])

    return data


def add_multiscale_features(data: pd.DataFrame, settings: ModeSettings) -> pd.DataFrame:
    data = data.copy()
    eps = 1e-12
    ret = data["ret_1"]
    log_close = data["log_close"]
    log_vol = data["log_volume"]

    confirmed_peak = ((log_close.shift(2) < log_close.shift(1)) & (log_close.shift(1) > log_close)).astype(float)
    confirmed_trough = ((log_close.shift(2) > log_close.shift(1)) & (log_close.shift(1) < log_close)).astype(float)
    data["confirmed_peak"] = confirmed_peak
    data["confirmed_trough"] = confirmed_trough

    for w in settings.scales:
        prefix = f"s{w}"
        if w == 1:
            data[f"{prefix}_ret_sum"] = ret
            data[f"{prefix}_ret_mean"] = ret
            data[f"{prefix}_ret_std"] = ret.rolling(3).std()
        else:
            data[f"{prefix}_ret_sum"] = log_close - log_close.shift(w)
            data[f"{prefix}_ret_mean"] = ret.rolling(w, min_periods=w).mean()
            data[f"{prefix}_ret_std"] = ret.rolling(w, min_periods=w).std()
        data[f"{prefix}_ret_min"] = ret.rolling(max(w, 2), min_periods=max(w, 2)).min()
        data[f"{prefix}_ret_max"] = ret.rolling(max(w, 2), min_periods=max(w, 2)).max()
        data[f"{prefix}_trend_slope"] = rolling_slope(log_close, max(w, 3))
        data[f"{prefix}_trend_r2"] = rolling_r2(log_close, max(w, 4))
        roll_max = log_close.rolling(max(w, 2), min_periods=max(w, 2)).max()
        roll_min = log_close.rolling(max(w, 2), min_periods=max(w, 2)).min()
        data[f"{prefix}_drawdown"] = log_close - roll_max
        data[f"{prefix}_bounce"] = log_close - roll_min
        data[f"{prefix}_range"] = roll_max - roll_min
        ma = log_close.rolling(max(w, 2), min_periods=max(w, 2)).mean()
        data[f"{prefix}_ma_gap"] = log_close - ma
        data[f"{prefix}_range_mean"] = data["high_low_range"].rolling(max(w, 2), min_periods=max(w, 2)).mean()
        vol_mean = log_vol.rolling(max(w, 2), min_periods=max(w, 2)).mean()
        vol_std = log_vol.rolling(max(w, 2), min_periods=max(w, 2)).std()
        data[f"{prefix}_volume_z"] = (log_vol - vol_mean) / (vol_std + eps)
        data[f"{prefix}_peak_count"] = confirmed_peak.rolling(max(w, 2), min_periods=max(w, 2)).sum()
        data[f"{prefix}_trough_count"] = confirmed_trough.rolling(max(w, 2), min_periods=max(w, 2)).sum()
        data[f"{prefix}_swing_density"] = (data[f"{prefix}_peak_count"] + data[f"{prefix}_trough_count"]) / max(w, 1)
        data[f"{prefix}_age_since_high"] = rolling_arg_age(log_close, max(w, 2), "max")
        data[f"{prefix}_age_since_low"] = rolling_arg_age(log_close, max(w, 2), "min")
        # Haar-like decomposition: recent half minus earlier half.
        if w >= 4:
            half = w // 2
            recent = ret.rolling(half, min_periods=half).sum()
            earlier = ret.shift(half).rolling(half, min_periods=half).sum()
            data[f"{prefix}_haar_recent_minus_prior"] = recent - earlier
            data[f"{prefix}_energy"] = (ret * ret).rolling(w, min_periods=w).sum()
        else:
            data[f"{prefix}_haar_recent_minus_prior"] = np.nan
            data[f"{prefix}_energy"] = ret * ret

    # volatility compression relative to larger scales.
    for w in settings.scales:
        if w < 7:
            continue
        bigger = min([s for s in settings.scales if s >= w * 2] or [settings.scales[-1]])
        if bigger != w and f"s{bigger}_ret_std" in data.columns:
            data[f"s{w}_vol_compression"] = data[f"s{w}_ret_std"] / (data[f"s{bigger}_ret_std"] + eps)

    return data


def add_spectral_features(data: pd.DataFrame, settings: ModeSettings) -> pd.DataFrame:
    data = data.copy()
    if sliding_window_view is None:
        return data
    ret = data["ret_1"].to_numpy(dtype=float)
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    n = len(ret)
    eps = 1e-12

    for w in settings.spectral_windows:
        if w < 8 or n < w + 5:
            continue
        windows = sliding_window_view(ret, window_shape=w)
        # Remove per-window mean, taper edges slightly.
        win = windows - windows.mean(axis=1, keepdims=True)
        taper = np.hanning(w)
        if np.allclose(taper, 0):
            taper = np.ones(w)
        fft = np.fft.rfft(win * taper, axis=1)
        power = np.abs(fft) ** 2
        if power.shape[1] <= 2:
            continue
        p = power[:, 1:]  # skip DC
        total = p.sum(axis=1) + eps
        dom_idx = np.argmax(p, axis=1) + 1
        dom_power = p[np.arange(len(p)), dom_idx - 1]
        dom_period = w / dom_idx.astype(float)
        strength = dom_power / total
        prob = p / total[:, None]
        entropy = -np.sum(prob * np.log(prob + eps), axis=1) / np.log(prob.shape[1] + eps)
        coeff = fft[np.arange(len(fft)), dom_idx]
        phase = np.angle(coeff)
        start = w - 1
        prefix = f"s{w}"
        for name, arr in [
            ("dominant_period", dom_period),
            ("cycle_strength", strength),
            ("spectral_entropy", entropy),
            ("cycle_phase_sin", np.sin(phase)),
            ("cycle_phase_cos", np.cos(phase)),
        ]:
            col = np.full(n, np.nan)
            col[start:] = arr
            data[f"{prefix}_{name}"] = col
    return data


def add_shape_convolution_features(data: pd.DataFrame, settings: ModeSettings, seed: int) -> pd.DataFrame:
    """MiniROCKET-inspired random convolutional responses using only past returns.

    This is not a full MiniROCKET implementation. It creates cheap local-shape
    features by convolving lagged returns with random zero-mean kernels.
    """
    data = data.copy()
    rng = np.random.default_rng(seed)
    ret = data["ret_1"].fillna(0.0).to_numpy(dtype=float)
    n = len(ret)
    kernel_lengths = np.array([3, 5, 7, 11, 15, 21, 31])
    max_len = min(max(settings.scales), 365)
    usable_lengths = kernel_lengths[kernel_lengths <= max_len]
    if len(usable_lengths) == 0:
        return data

    for k in range(settings.n_random_shape_kernels):
        L = int(rng.choice(usable_lengths))
        weights = rng.normal(0, 1, L)
        weights = weights - weights.mean()
        norm = np.sqrt(np.sum(weights * weights))
        if norm == 0:
            continue
        weights = weights / norm
        response = np.full(n, np.nan)
        # response at t uses ret[t-L+1:t+1].
        for i in range(L - 1, n):
            response[i] = float(np.dot(ret[i - L + 1 : i + 1], weights))
        data[f"shape_k{k:03d}_L{L}_resp"] = response
        # A local PPV-like feature: how often response was positive recently.
        w = min(max(14, L * 3), 120)
        s = pd.Series(response, index=data.index)
        data[f"shape_k{k:03d}_L{L}_ppv"] = (s > 0).rolling(w, min_periods=w).mean()
    return data


def add_regime_flags(data: pd.DataFrame, settings: ModeSettings) -> pd.DataFrame:
    data = data.copy()
    eps = 1e-12
    # Scale-aware flags. Thresholds are simple and past-safe; learned regimes are added after split.
    for w in settings.scales:
        prefix = f"s{w}"
        if f"{prefix}_trend_slope" not in data.columns:
            continue
        vol = data.get(f"{prefix}_ret_std", pd.Series(np.nan, index=data.index)).abs()
        trend = data[f"{prefix}_trend_slope"]
        ret_sum = data.get(f"{prefix}_ret_sum", pd.Series(np.nan, index=data.index))
        drawdown = data.get(f"{prefix}_drawdown", pd.Series(np.nan, index=data.index))
        bounce = data.get(f"{prefix}_bounce", pd.Series(np.nan, index=data.index))
        cycle_strength = data.get(f"{prefix}_cycle_strength", pd.Series(np.nan, index=data.index)).fillna(0)
        entropy = data.get(f"{prefix}_spectral_entropy", pd.Series(np.nan, index=data.index)).fillna(1)
        comp = data.get(f"{prefix}_vol_compression", pd.Series(np.nan, index=data.index))
        scaled_move = vol * math.sqrt(max(w, 1))
        data[f"flag_{prefix}_trend_up"] = (trend > 0).astype(float)
        data[f"flag_{prefix}_trend_down"] = (trend < 0).astype(float)
        data[f"flag_{prefix}_strong_up"] = (ret_sum > 0.75 * scaled_move).astype(float)
        data[f"flag_{prefix}_strong_down"] = (ret_sum < -0.75 * scaled_move).astype(float)
        data[f"flag_{prefix}_deep_dip"] = (drawdown < -1.0 * scaled_move).astype(float)
        data[f"flag_{prefix}_bounce_from_low"] = (bounce > 0.75 * scaled_move).astype(float)
        data[f"flag_{prefix}_near_high"] = (drawdown > -0.25 * scaled_move).astype(float)
        data[f"flag_{prefix}_near_low"] = (bounce < 0.25 * scaled_move).astype(float)
        data[f"flag_{prefix}_cyclic_chop"] = ((cycle_strength > 0.25) & (entropy < 0.8) & (trend.abs() < vol / max(math.sqrt(w), 1))).astype(float)
        if f"{prefix}_vol_compression" in data.columns:
            data[f"flag_{prefix}_compression"] = (comp < 0.75).astype(float)
            data[f"flag_{prefix}_vol_expansion"] = (comp > 1.35).astype(float)
        else:
            data[f"flag_{prefix}_compression"] = 0.0
            data[f"flag_{prefix}_vol_expansion"] = 0.0

    # Named cross-scale/hypercategory flags.
    def col(name: str) -> pd.Series:
        if name in data.columns:
            return data[name].fillna(0.0)
        return pd.Series(0.0, index=data.index)

    pairs = [(3, 30), (3, 60), (7, 60), (7, 120), (14, 120), (30, 240)]
    for short, long in pairs:
        if short not in settings.scales or long not in settings.scales:
            continue
        data[f"hyper_s{short}_dip_inside_s{long}_uptrend"] = (
            (col(f"flag_s{short}_deep_dip") > 0) & (col(f"flag_s{long}_trend_up") > 0)
        ).astype(float)
        data[f"hyper_s{short}_dip_inside_s{long}_downtrend"] = (
            (col(f"flag_s{short}_deep_dip") > 0) & (col(f"flag_s{long}_trend_down") > 0)
        ).astype(float)
        data[f"hyper_s{short}_breakout_inside_s{long}_compression"] = (
            (col(f"flag_s{short}_near_high") > 0)
            & (col(f"flag_s{short}_strong_up") > 0)
            & (col(f"flag_s{long}_compression") > 0)
        ).astype(float)
        data[f"hyper_s{short}_panic_inside_s{long}_downtrend"] = (
            (col(f"flag_s{short}_strong_down") > 0)
            & (col(f"flag_s{short}_vol_expansion") > 0)
            & (col(f"flag_s{long}_trend_down") > 0)
        ).astype(float)
        data[f"cross_s{short}_s{long}_trend_agree"] = (
            np.sign(data.get(f"s{short}_trend_slope", 0)) == np.sign(data.get(f"s{long}_trend_slope", 0))
        ).astype(float)
        data[f"cross_s{short}_s{long}_trend_product"] = data.get(f"s{short}_trend_slope", 0) * data.get(f"s{long}_trend_slope", 0)
        if f"s{short}_cycle_phase_sin" in data.columns and f"s{long}_cycle_phase_sin" in data.columns:
            data[f"cross_s{short}_s{long}_phase_alignment"] = (
                data[f"s{short}_cycle_phase_sin"] * data[f"s{long}_cycle_phase_sin"]
                + data[f"s{short}_cycle_phase_cos"] * data[f"s{long}_cycle_phase_cos"]
            )

    # Generic event/setup flag used for active-set evaluation.
    hyper_cols = [c for c in data.columns if c.startswith("hyper_")]
    event_cols = [c for c in data.columns if c.startswith("flag_s") and any(k in c for k in ["deep_dip", "compression", "vol_expansion", "cyclic_chop", "strong_up", "strong_down"])]
    if hyper_cols or event_cols:
        data["any_structure_event"] = (data[hyper_cols + event_cols].fillna(0).sum(axis=1) > 0).astype(float)
    else:
        data["any_structure_event"] = 0.0
    return data


def add_targets(data: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    data = data.copy()
    log_close = data["log_close"].to_numpy(dtype=float)
    high_log = data["log_high"].to_numpy(dtype=float)
    low_log = data["log_low"].to_numpy(dtype=float)
    n = len(data)
    for h in horizons:
        target = np.full(n, np.nan)
        direction = np.full(n, np.nan)
        max_runup = np.full(n, np.nan)
        max_drawdown = np.full(n, np.nan)
        future_vol = np.full(n, np.nan)
        for i in range(n - h):
            future_close = log_close[i + h]
            target[i] = future_close - log_close[i]
            direction[i] = 1.0 if target[i] > 0 else 0.0
            future_high = np.nanmax(high_log[i + 1 : i + h + 1])
            future_low = np.nanmin(low_log[i + 1 : i + h + 1])
            max_runup[i] = future_high - log_close[i]
            max_drawdown[i] = future_low - log_close[i]
            if h >= 2:
                future_rets = np.diff(log_close[i : i + h + 1])
                future_vol[i] = np.nanstd(future_rets)
            else:
                future_vol[i] = abs(target[i])
        data[f"target_return_{h}"] = target
        data[f"target_direction_{h}"] = direction
        data[f"target_max_runup_{h}"] = max_runup
        data[f"target_max_drawdown_{h}"] = max_drawdown
        data[f"target_future_vol_{h}"] = future_vol
    return data


def engineer_features(main: pd.DataFrame, context: Dict[str, pd.DataFrame], settings: ModeSettings, seed: int) -> pd.DataFrame:
    data = add_basic_context_features(main, context)
    data = add_multiscale_features(data, settings)
    data = add_spectral_features(data, settings)
    data = add_shape_convolution_features(data, settings, seed)
    data = add_regime_flags(data, settings)
    data = add_targets(data, settings.horizons)
    before = len(data)
    target_cols = [f"target_return_{h}" for h in settings.horizons]
    # Keep rows with all target horizons and enough feature history.
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=target_cols + ["Close", "log_close"]).reset_index(drop=True)
    after = len(data)
    print(f"Rows before drop target/history NaNs: {before}; after: {after}")
    print(f"Columns after feature engineering: {len(data.columns)}")
    if after:
        print(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
    return data


# -----------------------------------------------------------------------------
# Split and feature preparation
# -----------------------------------------------------------------------------


@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def chronological_split(data: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15) -> DataSplit:
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = data.iloc[:n_train].copy().reset_index(drop=True)
    val = data.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test = data.iloc[n_train + n_val :].copy().reset_index(drop=True)
    return DataSplit(train=train, val=val, test=test)


def raw_exclusion_cols(data: pd.DataFrame) -> List[str]:
    base = {"Date", "Open", "High", "Low", "Close", "Volume"}
    for c in data.columns:
        if c.startswith("target_"):
            base.add(c)
    return sorted(base)


def numeric_feature_columns(data: pd.DataFrame) -> List[str]:
    exclude = set(raw_exclusion_cols(data))
    cols = []
    for c in data.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(data[c]):
            cols.append(c)
    return cols


def add_latent_regimes(split: DataSplit, feature_cols: List[str], settings: ModeSettings, seed: int) -> Tuple[DataSplit, List[str]]:
    candidates = [
        c
        for c in feature_cols
        if any(k in c for k in ["ret_sum", "ret_std", "trend_slope", "drawdown", "bounce", "cycle_strength", "spectral_entropy", "vol_compression", "range", "volume_z"])
    ]
    if len(candidates) < 5 or settings.n_latent_regimes <= 1:
        return split, feature_cols
    # Use a modest subset by variance to keep GMM stable.
    variances = split.train[candidates].replace([np.inf, -np.inf], np.nan).var(numeric_only=True).sort_values(ascending=False)
    selected = variances.head(min(50, len(variances))).index.tolist()
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    X_train = pipe.fit_transform(split.train[selected])
    n_components = min(settings.n_latent_regimes, max(2, len(split.train) // 300))
    if n_components < 2:
        return split, feature_cols
    try:
        gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=seed, reg_covar=1e-4, max_iter=200)
        gmm.fit(X_train)
        prob_train = gmm.predict_proba(X_train)
        prob_val = gmm.predict_proba(pipe.transform(split.val[selected]))
        prob_test = gmm.predict_proba(pipe.transform(split.test[selected]))
    except Exception:
        return split, feature_cols

    def _assign(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
        out = df.copy()
        hard = probs.argmax(axis=1)
        for k in range(probs.shape[1]):
            out[f"latent_regime_{k}_prob"] = probs[:, k]
            out[f"latent_regime_{k}_hard"] = (hard == k).astype(float)
        return out

    new_split = DataSplit(_assign(split.train, prob_train), _assign(split.val, prob_val), _assign(split.test, prob_test))
    new_cols = numeric_feature_columns(new_split.train)
    return new_split, new_cols


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@dataclass
class MetricRow:
    track: str
    model: str
    split: str
    horizon: int
    n_samples: int
    return_mae: float
    return_rmse: float
    price_mae: float
    price_rmse: float
    directional_accuracy: float
    return_correlation: float
    active_coverage: float
    active_directional_accuracy: float
    active_return_mae: float
    selected_by: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def directional_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    ysign = np.sign(y)
    psign = np.sign(p)
    mask = (ysign != 0) & (psign != 0) & np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(ysign[mask] == psign[mask]))


def correlation(y: np.ndarray, p: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 3:
        return float("nan")
    yy = y[mask]
    pp = p[mask]
    if np.std(yy) == 0 or np.std(pp) == 0:
        return float("nan")
    return float(np.corrcoef(yy, pp)[0, 1])


def evaluate_predictions(
    df: pd.DataFrame,
    pred_return: np.ndarray,
    horizon: int,
    track: str,
    model: str,
    split_name: str,
    selected_by: str = "",
    active_mask: Optional[np.ndarray] = None,
) -> MetricRow:
    y = df[f"target_return_{horizon}"].to_numpy(dtype=float)
    pred = np.asarray(pred_return, dtype=float).reshape(-1)
    n = min(len(y), len(pred))
    y = y[:n]
    pred = pred[:n]
    close = df["Close"].to_numpy(dtype=float)[:n]
    true_price = close * np.exp(y)
    pred_price = close * np.exp(pred)
    mask = np.isfinite(y) & np.isfinite(pred) & np.isfinite(close)
    if mask.sum() == 0:
        mae = rms = pmae = prms = da = corr = float("nan")
    else:
        mae = float(mean_absolute_error(y[mask], pred[mask]))
        rms = rmse(y[mask], pred[mask])
        pmae = float(mean_absolute_error(true_price[mask], pred_price[mask]))
        prms = rmse(true_price[mask], pred_price[mask])
        da = directional_accuracy(y[mask], pred[mask])
        corr = correlation(y[mask], pred[mask])

    if active_mask is None:
        active_mask = np.abs(pred) > np.nanmedian(np.abs(pred[np.isfinite(pred)])) if np.isfinite(pred).any() else np.zeros_like(pred, dtype=bool)
    active_mask = np.asarray(active_mask, dtype=bool)[:n] & mask
    coverage = float(active_mask.mean()) if n else float("nan")
    if active_mask.sum() > 0:
        ada = directional_accuracy(y[active_mask], pred[active_mask])
        amae = float(mean_absolute_error(y[active_mask], pred[active_mask]))
    else:
        ada = float("nan")
        amae = float("nan")
    return MetricRow(
        track=track,
        model=model,
        split=split_name,
        horizon=horizon,
        n_samples=int(mask.sum()),
        return_mae=mae,
        return_rmse=rms,
        price_mae=pmae,
        price_rmse=prms,
        directional_accuracy=da,
        return_correlation=corr,
        active_coverage=coverage,
        active_directional_accuracy=ada,
        active_return_mae=amae,
        selected_by=selected_by,
    )


def choose_active_threshold(val_y: np.ndarray, val_pred: np.ndarray, min_coverage: float = 0.08) -> float:
    conf = np.abs(val_pred)
    mask = np.isfinite(val_y) & np.isfinite(val_pred)
    if mask.sum() < 20:
        return float("inf")
    conf = conf[mask]
    y = val_y[mask]
    p = val_pred[mask]
    best = (0.0, -1.0)  # threshold, score
    for q in np.linspace(0.50, 0.92, 18):
        thr = float(np.quantile(conf, q))
        active = np.abs(p) >= thr
        coverage = active.mean()
        if coverage < min_coverage or active.sum() < 20:
            continue
        da = directional_accuracy(y[active], p[active])
        mae = mean_absolute_error(y[active], p[active])
        score = (0.0 if np.isnan(da) else da) - 0.25 * mae
        if score > best[1]:
            best = (thr, score)
    if best[1] < 0:
        return float(np.quantile(conf, 0.75))
    return best[0]


# -----------------------------------------------------------------------------
# Modeling helpers
# -----------------------------------------------------------------------------


def clean_xy(df: pd.DataFrame, feature_cols: List[str], horizon: int) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[f"target_return_{horizon}"].to_numpy(dtype=float)
    return X, y


def model_pipeline(kind: str, seed: int):
    if kind == "ridge":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
                ("model", Ridge(alpha=10.0, random_state=seed)),
            ]
        )
    if kind == "elastic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
                ("model", ElasticNet(alpha=0.0005, l1_ratio=0.20, random_state=seed, max_iter=5000)),
            ]
        )
    if kind == "extra_trees":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=60,
                        max_depth=8,
                        min_samples_leaf=12,
                        max_features=0.35,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if kind == "extra_trees_deeper":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_leaf=8,
                        max_features=0.45,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if kind == "hist_gb":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.035,
                        max_iter=80,
                        max_leaf_nodes=24,
                        l2_regularization=0.03,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(kind)


def classifier_pipeline(seed: int):
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=90,
                    max_depth=9,
                    min_samples_leaf=12,
                    max_features=0.45,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def top_correlated_features(train: pd.DataFrame, feature_cols: List[str], target_col: str, n: int) -> List[str]:
    y = train[target_col].to_numpy(dtype=float)
    scores = []
    for c in feature_cols:
        x = train[c].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 30 or np.nanstd(x[mask]) == 0:
            continue
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        if np.isfinite(corr):
            scores.append((abs(float(corr)), c))
    scores.sort(reverse=True)
    return [c for _, c in scores[:n]]


# -----------------------------------------------------------------------------
# Formula / hyperfunction layer
# -----------------------------------------------------------------------------


def standardized_triplet(train: pd.Series, val: pd.Series, test: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = float(np.nanmedian(train))
    sd = float(np.nanstd(train))
    if not np.isfinite(sd) or sd < 1e-9:
        sd = 1.0
    def z(s: pd.Series) -> np.ndarray:
        return np.nan_to_num((s.to_numpy(dtype=float) - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return z(train), z(val), z(test)


def generate_formula_layer(
    split: DataSplit,
    feature_cols: List[str],
    horizon: int,
    settings: ModeSettings,
    seed: int,
) -> Tuple[DataSplit, List[str], List[str]]:
    rng = np.random.default_rng(seed)
    target = f"target_return_{horizon}"
    # Base continuous features chosen by train correlation.
    base = top_correlated_features(split.train, feature_cols, target, settings.n_formula_base_features)
    category_candidates = [c for c in feature_cols if c.startswith("flag_") or c.startswith("hyper_") or c.startswith("latent_regime_") or c.startswith("cross_")]
    # Prefer categories with both nonzero variance and some target association.
    cats = top_correlated_features(split.train, category_candidates, target, settings.n_formula_category_features)
    if not base:
        return split, feature_cols, []

    formula_train: Dict[str, np.ndarray] = {}
    formula_val: Dict[str, np.ndarray] = {}
    formula_test: Dict[str, np.ndarray] = {}
    formula_desc: Dict[str, str] = {}

    z_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for c in base:
        z_cache[c] = standardized_triplet(split.train[c], split.val[c], split.test[c])

    # Unary transformations.
    for c in base:
        tr, va, te = z_cache[c]
        for name, ftr, fva, fte in [
            (f"form_tanh__{c}", np.tanh(tr), np.tanh(va), np.tanh(te)),
            (f"form_abs__{c}", np.abs(tr), np.abs(va), np.abs(te)),
            (f"form_sq__{c}", np.clip(tr * tr, 0, 25), np.clip(va * va, 0, 25), np.clip(te * te, 0, 25)),
        ]:
            formula_train[name] = ftr
            formula_val[name] = fva
            formula_test[name] = fte
            formula_desc[name] = name.replace("form_", "")

    # Feature x category interactions: this is the main category-conditioned formula layer.
    for c in base[: min(len(base), 35)]:
        tr, va, te = z_cache[c]
        for cat in cats[: min(len(cats), 30)]:
            ctr = np.nan_to_num(split.train[cat].to_numpy(dtype=float), nan=0.0)
            cva = np.nan_to_num(split.val[cat].to_numpy(dtype=float), nan=0.0)
            cte = np.nan_to_num(split.test[cat].to_numpy(dtype=float), nan=0.0)
            name = f"form_interact__{c}__X__{cat}"
            formula_train[name] = tr * ctr
            formula_val[name] = va * cva
            formula_test[name] = te * cte
            formula_desc[name] = f"z({c}) * {cat}"

    # Cross-scale hand-built differences/products for similar features across scales.
    scales = list(settings.scales)
    for short, long in [(1, 7), (2, 14), (3, 30), (7, 60), (14, 120), (30, 240), (60, 240)]:
        if short not in scales or long not in scales:
            continue
        for stem in ["ret_sum", "trend_slope", "drawdown", "bounce", "ret_std", "cycle_strength"]:
            a = f"s{short}_{stem}"
            b = f"s{long}_{stem}"
            if a not in feature_cols or b not in feature_cols:
                continue
            za = standardized_triplet(split.train[a], split.val[a], split.test[a])
            zb = standardized_triplet(split.train[b], split.val[b], split.test[b])
            name1 = f"form_scale_diff__{a}__minus__{b}"
            name2 = f"form_scale_prod__{a}__times__{b}"
            formula_train[name1] = za[0] - zb[0]
            formula_val[name1] = za[1] - zb[1]
            formula_test[name1] = za[2] - zb[2]
            formula_desc[name1] = f"z({a}) - z({b})"
            formula_train[name2] = np.clip(za[0] * zb[0], -25, 25)
            formula_val[name2] = np.clip(za[1] * zb[1], -25, 25)
            formula_test[name2] = np.clip(za[2] * zb[2], -25, 25)
            formula_desc[name2] = f"z({a}) * z({b})"

    # Decayed onion summaries: learn whether short or long scales dominate.
    for tau in [3, 7, 14, 30, 60, 120, 240]:
        cols = [f"s{s}_ret_sum" for s in settings.scales if f"s{s}_ret_sum" in feature_cols]
        if len(cols) < 3:
            continue
        weights = np.array([math.exp(-s / tau) for s in settings.scales if f"s{s}_ret_sum" in feature_cols], dtype=float)
        weights = weights / (weights.sum() + 1e-12)
        arrays = []
        for c in cols:
            arrays.append(standardized_triplet(split.train[c], split.val[c], split.test[c]))
        name = f"form_decay_weighted_ret_tau{tau}"
        formula_train[name] = np.sum([weights[i] * arrays[i][0] for i in range(len(arrays))], axis=0)
        formula_val[name] = np.sum([weights[i] * arrays[i][1] for i in range(len(arrays))], axis=0)
        formula_test[name] = np.sum([weights[i] * arrays[i][2] for i in range(len(arrays))], axis=0)
        formula_desc[name] = f"decayed weighted sum of multi-scale returns, tau={tau}"

    if not formula_train:
        return split, feature_cols, []

    y = split.train[target].to_numpy(dtype=float)
    scored = []
    for name, arr in formula_train.items():
        mask = np.isfinite(y) & np.isfinite(arr)
        if mask.sum() < 50 or np.std(arr[mask]) == 0:
            continue
        corr = np.corrcoef(y[mask], arr[mask])[0, 1]
        if np.isfinite(corr):
            scored.append((abs(float(corr)), name))
    scored.sort(reverse=True)
    selected = [name for _, name in scored[: settings.max_selected_formulas]]

    def _append(df: pd.DataFrame, store: Dict[str, np.ndarray]) -> pd.DataFrame:
        out = df.copy()
        if selected:
            add = pd.DataFrame({name: store[name] for name in selected}, index=out.index)
            out = pd.concat([out, add], axis=1)
        return out

    new_split = DataSplit(_append(split.train, formula_train), _append(split.val, formula_val), _append(split.test, formula_test))
    new_cols = feature_cols + selected
    descriptions = [f"{name}: {formula_desc.get(name, name)}" for name in selected]
    return new_split, new_cols, descriptions


# -----------------------------------------------------------------------------
# Baselines, experts, and gated stack
# -----------------------------------------------------------------------------


def baseline_predictions(df: pd.DataFrame, horizon: int) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    n = len(df)
    preds["zero_return_persistence"] = np.zeros(n)
    for w in [3, 7, 14, 30, 60, 90, 120]:
        col = f"s{w}_ret_mean"
        if col in df.columns:
            preds[f"ret_mean_{w}_scaled_to_h"] = np.nan_to_num(df[col].to_numpy(dtype=float) * horizon, nan=0.0)
    if f"s{horizon}_ret_sum" in df.columns:
        preds[f"trailing_{horizon}d_momentum"] = np.nan_to_num(df[f"s{horizon}_ret_sum"].to_numpy(dtype=float), nan=0.0)
    # simple dip-bounce baseline: if short dip in larger uptrend, predict bounce of a fraction of recent vol.
    for short, long in [(3, 60), (7, 120)]:
        flag = f"hyper_s{short}_dip_inside_s{long}_uptrend"
        vol = f"s{short}_ret_std"
        if flag in df.columns and vol in df.columns:
            preds[f"simple_dip_bounce_s{short}_s{long}"] = np.nan_to_num(df[flag].to_numpy(dtype=float) * df[vol].to_numpy(dtype=float) * math.sqrt(max(horizon, 1)), nan=0.0)
    return preds


def train_return_models(
    split: DataSplit,
    feature_cols: List[str],
    horizon: int,
    seed: int,
    settings: ModeSettings,
) -> Tuple[List[MetricRow], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, object]]:
    rows: List[MetricRow] = []
    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    fitted: Dict[str, object] = {}
    X_train, y_train = clean_xy(split.train, feature_cols, horizon)
    X_val, y_val = clean_xy(split.val, feature_cols, horizon)
    X_test, y_test = clean_xy(split.test, feature_cols, horizon)

    specs = ["ridge", "elastic", "extra_trees"]
    # HistGradientBoosting is useful, but slower with wide formula features,
    # so skip it in quick mode where include_neural is False.
    if settings.include_neural:
        specs.append("hist_gb")
    if settings.max_selected_formulas > 120:
        specs.append("extra_trees_deeper")
    for kind in specs:
        print(f"Training global return model: h={horizon} | {kind}")
        try:
            model = model_pipeline(kind, seed)
            model.fit(X_train, y_train)
            val_pred = np.asarray(model.predict(X_val), dtype=float).reshape(-1)
            test_pred = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
            val_mae = mean_absolute_error(y_val, val_pred)
            row = evaluate_predictions(split.test, test_pred, horizon, "global_model", kind, "test", selected_by=f"val_mae={val_mae:.6f}")
            rows.append(row)
            val_preds[kind] = val_pred
            test_preds[kind] = test_pred
            fitted[kind] = model
            print(f"  val MAE={val_mae:.6f} | test MAE={row.return_mae:.6f} | dir={row.directional_accuracy:.3f}")
        except Exception as exc:
            print(f"  skipped {kind}: {exc}")
    return rows, val_preds, test_preds, fitted


def train_direction_classifier_as_return(
    split: DataSplit,
    feature_cols: List[str],
    horizon: int,
    seed: int,
) -> Tuple[List[MetricRow], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rows: List[MetricRow] = []
    X_train, y_train_ret = clean_xy(split.train, feature_cols, horizon)
    X_val, y_val_ret = clean_xy(split.val, feature_cols, horizon)
    X_test, y_test_ret = clean_xy(split.test, feature_cols, horizon)
    y_train_dir = (y_train_ret > 0).astype(int)
    if len(np.unique(y_train_dir)) < 2:
        return rows, {}, {}
    model = classifier_pipeline(seed)
    model.fit(X_train, y_train_dir)
    prob_val = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]
    signed_val = 2 * prob_val - 1
    signed_test = 2 * prob_test - 1
    # Magnitude calibration chosen on validation only.
    mag_base = np.nanmedian(np.abs(y_train_ret))
    if not np.isfinite(mag_base) or mag_base <= 0:
        mag_base = 0.01
    candidates = mag_base * np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])
    best_mag = candidates[0]
    best_mae = float("inf")
    for mag in candidates:
        val_pred = signed_val * mag
        mae = mean_absolute_error(y_val_ret, val_pred)
        if mae < best_mae:
            best_mae = mae
            best_mag = float(mag)
    val_pred = signed_val * best_mag
    test_pred = signed_test * best_mag
    thr = choose_active_threshold(y_val_ret, val_pred)
    active = np.abs(test_pred) >= thr
    row = evaluate_predictions(
        split.test,
        test_pred,
        horizon,
        "direction_probability",
        "event_direction_classifier",
        "test",
        selected_by=f"val_mae={best_mae:.6f};mag={best_mag:.6f};thr={thr:.6f}",
        active_mask=active,
    )
    rows.append(row)
    print(f"Training direction-probability model: h={horizon} | val MAE={best_mae:.6f} | test MAE={row.return_mae:.6f} | dir={row.directional_accuracy:.3f}")
    return rows, {"event_direction_classifier": val_pred}, {"event_direction_classifier": test_pred}


def select_specialist_flags(train: pd.DataFrame, feature_cols: List[str], settings: ModeSettings) -> List[str]:
    candidates = [c for c in feature_cols if c.startswith("hyper_") or c in ["any_structure_event"]]
    candidates += [c for c in feature_cols if c.startswith("flag_s") and any(k in c for k in ["deep_dip", "compression", "vol_expansion", "cyclic_chop", "strong_up", "strong_down"])]
    out = []
    for c in candidates:
        vals = train[c].fillna(0).to_numpy(dtype=float)
        count = int((vals > 0.5).sum())
        if count >= settings.min_specialist_train:
            out.append(c)
    # Keep specialist count manageable and favor interpretable hypercategories.
    limit = 5 if not settings.include_neural else (14 if settings.max_selected_formulas <= 120 else 22)
    out = sorted(out, key=lambda x: (0 if x.startswith("hyper_") else 1, x))[:limit]
    return out


def train_specialist_experts(
    split: DataSplit,
    feature_cols: List[str],
    horizon: int,
    seed: int,
    settings: ModeSettings,
) -> Tuple[List[MetricRow], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rows: List[MetricRow] = []
    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    flags = select_specialist_flags(split.train, feature_cols, settings)
    if not flags:
        return rows, val_preds, test_preds
    X_train, y_train = clean_xy(split.train, feature_cols, horizon)
    X_val, y_val = clean_xy(split.val, feature_cols, horizon)
    X_test, y_test = clean_xy(split.test, feature_cols, horizon)
    # Fallback global specialist.
    fallback = model_pipeline("extra_trees", seed + 101)
    fallback.fit(X_train, y_train)
    fallback_val = fallback.predict(X_val)
    fallback_test = fallback.predict(X_test)

    specialist_val_matrix = []
    specialist_test_matrix = []
    weights = []
    names = []
    for idx, flag in enumerate(flags):
        mask = split.train[flag].fillna(0).to_numpy(dtype=float) > 0.5
        if mask.sum() < settings.min_specialist_train:
            continue
        try:
            model = model_pipeline("extra_trees", seed + 1000 + idx)
            model.fit(X_train.iloc[mask], y_train[mask])
            val_p = np.asarray(model.predict(X_val), dtype=float).reshape(-1)
            test_p = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
            # Validation weight: trust specialists that perform well on their own event rows.
            val_event = split.val[flag].fillna(0).to_numpy(dtype=float) > 0.5
            if val_event.sum() >= settings.min_event_val:
                spec_mae = mean_absolute_error(y_val[val_event], val_p[val_event])
            else:
                spec_mae = mean_absolute_error(y_val, val_p)
            weights.append(1.0 / (spec_mae + 1e-6))
            specialist_val_matrix.append(val_p)
            specialist_test_matrix.append(test_p)
            names.append(flag)
        except Exception:
            continue
    if not specialist_val_matrix:
        return rows, val_preds, test_preds
    W = np.asarray(weights, dtype=float)
    W = W / W.sum()
    val_stack = np.vstack(specialist_val_matrix)
    test_stack = np.vstack(specialist_test_matrix)
    # Gate by active flag: specialist output matters most when its flag is active.
    val_gate = []
    test_gate = []
    for name in names:
        val_gate.append(split.val[name].fillna(0).to_numpy(dtype=float))
        test_gate.append(split.test[name].fillna(0).to_numpy(dtype=float))
    val_gate = np.vstack(val_gate)
    test_gate = np.vstack(test_gate)
    val_gate_weighted = val_gate * W[:, None]
    test_gate_weighted = test_gate * W[:, None]
    val_denom = val_gate_weighted.sum(axis=0)
    test_denom = test_gate_weighted.sum(axis=0)
    val_pred = np.where(val_denom > 0, (val_stack * val_gate_weighted).sum(axis=0) / (val_denom + 1e-12), fallback_val)
    test_pred = np.where(test_denom > 0, (test_stack * test_gate_weighted).sum(axis=0) / (test_denom + 1e-12), fallback_test)
    val_mae = mean_absolute_error(y_val, val_pred)
    thr = choose_active_threshold(y_val, val_pred, settings.min_active_coverage)
    active = np.abs(test_pred) >= thr
    row = evaluate_predictions(
        split.test,
        test_pred,
        horizon,
        "mixture_specialists",
        "flag_gated_specialist_ensemble",
        "test",
        selected_by=f"val_mae={val_mae:.6f};n_specialists={len(names)};thr={thr:.6f}",
        active_mask=active,
    )
    rows.append(row)
    val_preds["flag_gated_specialist_ensemble"] = val_pred
    test_preds["flag_gated_specialist_ensemble"] = test_pred
    print(f"Training specialist ensemble: h={horizon} | specialists={len(names)} | val MAE={val_mae:.6f} | test MAE={row.return_mae:.6f} | dir={row.directional_accuracy:.3f}")
    return rows, val_preds, test_preds


def train_optional_neural_expert(
    split: DataSplit,
    feature_cols: List[str],
    horizon: int,
    seed: int,
    settings: ModeSettings,
) -> Tuple[List[MetricRow], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if not settings.include_neural:
        return [], {}, {}
    try:
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras
        from keras import layers
    except Exception as exc:
        print(f"Skipping neural expert: {exc}")
        return [], {}, {}
    # Keep neural input small and correlated to reduce overfitting.
    selected = top_correlated_features(split.train, feature_cols, f"target_return_{horizon}", 80)
    if len(selected) < 10:
        return [], {}, {}
    lookback = 60 if horizon <= 7 else 120
    if len(split.train) <= lookback + 50 or len(split.val) <= lookback + 20:
        return [], {}, {}

    def make_windows(train_df: pd.DataFrame, fit_scaler: bool, scaler: Optional[Pipeline] = None):
        Xdf = train_df[selected].replace([np.inf, -np.inf], np.nan)
        if scaler is None:
            scaler = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())])
            Xall = scaler.fit_transform(Xdf) if fit_scaler else scaler.transform(Xdf)
        else:
            Xall = scaler.transform(Xdf)
        y = train_df[f"target_return_{horizon}"].to_numpy(dtype=float)
        d = (y > 0).astype(float)
        Xs, ys, ds, rows = [], [], [], []
        for i in range(lookback - 1, len(train_df)):
            Xs.append(Xall[i - lookback + 1 : i + 1])
            ys.append(y[i])
            ds.append(d[i])
            rows.append(i)
        return np.asarray(Xs, dtype="float32"), np.asarray(ys, dtype="float32"), np.asarray(ds, dtype="float32"), np.asarray(rows), scaler

    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass
    X_train, y_train, d_train, _, scaler = make_windows(split.train, True)
    X_val, y_val, d_val, val_rows, _ = make_windows(split.val, False, scaler)
    X_test, y_test, d_test, test_rows, _ = make_windows(split.test, False, scaler)
    if len(X_train) == 0:
        return [], {}, {}

    inp = layers.Input(shape=X_train.shape[1:])
    x = layers.Conv1D(32, 5, padding="causal", activation="relu")(inp)
    x = layers.Dropout(0.15)(x)
    x = layers.LSTM(48)(x)
    x = layers.Dropout(0.20)(x)
    x = layers.Dense(32, activation="relu")(x)
    ret_out = layers.Dense(1, name="ret")(x)
    dir_out = layers.Dense(1, activation="sigmoid", name="dir")(x)
    model = keras.Model(inp, [ret_out, dir_out])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={"ret": "huber", "dir": "binary_crossentropy"},
        loss_weights={"ret": 1.0, "dir": 0.25},
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_ret_loss", mode="min", patience=settings.neural_patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_ret_loss", mode="min", factor=0.5, patience=max(3, settings.neural_patience // 2), min_lr=1e-5),
    ]
    print(f"Training neural sequence expert: h={horizon} | lookback={lookback}")
    try:
        model.fit(
            X_train,
            {"ret": y_train, "dir": d_train},
            validation_data=(X_val, {"ret": y_val, "dir": d_val}),
            epochs=settings.neural_epochs,
            batch_size=32,
            verbose=0,
            callbacks=callbacks,
        )
        val_pred = model.predict(X_val, verbose=0)[0].reshape(-1)
        test_pred = model.predict(X_test, verbose=0)[0].reshape(-1)
        full_val = np.zeros(len(split.val))
        full_test = np.zeros(len(split.test))
        full_val[val_rows] = val_pred
        full_test[test_rows] = test_pred
        val_mae = mean_absolute_error(split.val.iloc[val_rows][f"target_return_{horizon}"], val_pred)
        thr = choose_active_threshold(split.val[f"target_return_{horizon}"].to_numpy(dtype=float), full_val, settings.min_active_coverage)
        row = evaluate_predictions(split.test, full_test, horizon, "neural_sequence", "conv_lstm_multitask_expert", "test", selected_by=f"val_mae={val_mae:.6f};thr={thr:.6f}", active_mask=np.abs(full_test) >= thr)
        print(f"  neural val MAE={val_mae:.6f} | test MAE={row.return_mae:.6f} | dir={row.directional_accuracy:.3f}")
        return [row], {"conv_lstm_multitask_expert": full_val}, {"conv_lstm_multitask_expert": full_test}
    except Exception as exc:
        print(f"  neural expert failed/skipped: {exc}")
        return [], {}, {}


def train_gated_meta_ensemble(
    split: DataSplit,
    base_val: Dict[str, np.ndarray],
    base_test: Dict[str, np.ndarray],
    feature_cols: List[str],
    horizon: int,
    seed: int,
    settings: ModeSettings,
) -> Tuple[List[MetricRow], Dict[str, np.ndarray]]:
    """Final hierarchy combiner.

    Instead of letting a high-capacity meta-model overfit a small validation set,
    this gate uses validation performance to weight base experts. This is still a
    hierarchy because formula/category/specialist/neural outputs all feed this
    final layer, but the combiner is deliberately conservative.
    """
    rows: List[MetricRow] = []
    if len(base_val) < 2:
        return rows, {}
    y_val = split.val[f"target_return_{horizon}"].to_numpy(dtype=float)

    scored = []
    for name, pred in base_val.items():
        pred = np.asarray(pred, dtype=float).reshape(-1)
        if len(pred) != len(y_val):
            continue
        mask = np.isfinite(y_val) & np.isfinite(pred)
        if mask.sum() < 30:
            continue
        mae = mean_absolute_error(y_val[mask], pred[mask])
        if np.isfinite(mae):
            scored.append((mae, name))
    if len(scored) < 2:
        return rows, {}
    scored.sort()
    # Keep the strongest validation experts, but preserve diversity by allowing
    # baselines, global models, specialists, and direction models through if they
    # validate well.
    top = scored[: min(8, len(scored))]
    names = [name for _, name in top]
    maes = np.array([mae for mae, _ in top], dtype=float)
    weights = 1.0 / np.square(maes + 1e-6)
    weights = weights / weights.sum()

    val_mat = np.vstack([base_val[name] for name in names])
    test_mat = np.vstack([base_test[name] for name in names])
    val_pred = np.sum(weights[:, None] * val_mat, axis=0)
    test_pred = np.sum(weights[:, None] * test_mat, axis=0)

    # Optional conservative shrinkage toward zero-return persistence. The shrink
    # coefficient is chosen on validation only.
    best_alpha = 1.0
    best_mae = float("inf")
    for alpha in np.linspace(0.25, 1.0, 16):
        p = alpha * val_pred
        mae = mean_absolute_error(y_val, p)
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    val_pred = best_alpha * val_pred
    test_pred = best_alpha * test_pred

    thr = choose_active_threshold(y_val, val_pred, settings.min_active_coverage)
    row = evaluate_predictions(
        split.test,
        test_pred,
        horizon,
        "stacked_hierarchy",
        f"validation_weighted_blend_{len(names)}_experts",
        "test",
        selected_by=f"val_mae={best_mae:.6f};alpha={best_alpha:.2f};thr={thr:.6f};experts={','.join(names)}",
        active_mask=np.abs(test_pred) >= thr,
    )
    rows.append(row)
    print(
        f"Training final validation-weighted gate: h={horizon} | experts={len(names)} | "
        f"val MAE={best_mae:.6f} | test MAE={row.return_mae:.6f} | dir={row.directional_accuracy:.3f}"
    )
    return rows, {"final_gated_hierarchy": test_pred}


# -----------------------------------------------------------------------------
# Ablation and walk-forward diagnostics
# -----------------------------------------------------------------------------


def feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    groups = {
        "base_ohlcv_context": [],
        "multiscale_trend_vol": [],
        "spectral_cycle": [],
        "shape_convolution": [],
        "regime_hypercategory": [],
        "formula_interaction": [],
    }
    for c in feature_cols:
        if c.startswith("form_"):
            groups["formula_interaction"].append(c)
        elif c.startswith("shape_"):
            groups["shape_convolution"].append(c)
        elif "cycle" in c or "spectral" in c or "dominant_period" in c:
            groups["spectral_cycle"].append(c)
        elif c.startswith("flag_") or c.startswith("hyper_") or c.startswith("cross_") or c.startswith("latent_regime") or c == "any_structure_event":
            groups["regime_hypercategory"].append(c)
        elif c.startswith("s") and any(k in c for k in ["trend", "ret_", "drawdown", "bounce", "vol", "range", "ma_gap", "peak", "trough"]):
            groups["multiscale_trend_vol"].append(c)
        else:
            groups["base_ohlcv_context"].append(c)
    return {k: v for k, v in groups.items() if v}


def run_feature_ablation(split: DataSplit, feature_cols: List[str], horizon: int, seed: int) -> pd.DataFrame:
    groups = feature_groups(feature_cols)
    rows = []
    all_model = model_pipeline("extra_trees", seed)
    Xtr, ytr = clean_xy(split.train, feature_cols, horizon)
    Xv, yv = clean_xy(split.val, feature_cols, horizon)
    Xt, yt = clean_xy(split.test, feature_cols, horizon)
    try:
        all_model.fit(Xtr, ytr)
        pred = all_model.predict(Xt)
        rows.append({"feature_group": "ALL", "n_features": len(feature_cols), "test_return_mae": mean_absolute_error(yt, pred), "test_directional_accuracy": directional_accuracy(yt, pred)})
    except Exception:
        pass
    for g, cols in groups.items():
        if len(cols) < 2:
            continue
        try:
            m = model_pipeline("extra_trees", seed + len(rows))
            m.fit(split.train[cols], ytr)
            p = m.predict(split.test[cols])
            rows.append({"feature_group": g, "n_features": len(cols), "test_return_mae": mean_absolute_error(yt, p), "test_directional_accuracy": directional_accuracy(yt, p)})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("test_return_mae") if rows else pd.DataFrame()


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def run_one_horizon(
    split: DataSplit,
    feature_cols: List[str],
    horizon: int,
    settings: ModeSettings,
    seed: int,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    section(f"Horizon {horizon} days")
    rows: List[MetricRow] = []
    base_val: Dict[str, np.ndarray] = {}
    base_test: Dict[str, np.ndarray] = {}

    # Baselines first.
    for name, pred in baseline_predictions(split.test, horizon).items():
        row = evaluate_predictions(split.test, pred, horizon, "baseline", name, "test", selected_by="fixed_baseline", active_mask=np.abs(pred) > 0)
        rows.append(row)
    for name, pred in baseline_predictions(split.val, horizon).items():
        base_val[f"baseline_{name}"] = pred
    for name, pred in baseline_predictions(split.test, horizon).items():
        base_test[f"baseline_{name}"] = pred

    # Formula layer is generated per horizon and then feeds later models.
    split_h, feat_h, formulas = generate_formula_layer(split, feature_cols, horizon, settings, seed + horizon * 13)
    if formulas:
        formula_path = out_dir / f"selected_formulas_h{horizon}.txt"
        formula_path.write_text("\n".join(formulas[:300]), encoding="utf-8")
        print(f"Generated/selected formula features: {len(formulas)}")
    else:
        print("No formula features selected.")

    # Global return models on augmented formula/category/stat features.
    model_rows, val_preds, test_preds, _ = train_return_models(split_h, feat_h, horizon, seed + horizon * 101, settings)
    rows.extend(model_rows)
    base_val.update({f"global_{k}": v for k, v in val_preds.items()})
    base_test.update({f"global_{k}": v for k, v in test_preds.items()})

    # Event direction classifier as return via calibrated probability magnitude.
    dir_rows, dir_val, dir_test = train_direction_classifier_as_return(split_h, feat_h, horizon, seed + horizon * 211)
    rows.extend(dir_rows)
    base_val.update({f"dir_{k}": v for k, v in dir_val.items()})
    base_test.update({f"dir_{k}": v for k, v in dir_test.items()})

    # Specialist mixture using flags/hypercategories.
    spec_rows, spec_val, spec_test = train_specialist_experts(split_h, feat_h, horizon, seed + horizon * 307, settings)
    rows.extend(spec_rows)
    base_val.update({f"specialist_{k}": v for k, v in spec_val.items()})
    base_test.update({f"specialist_{k}": v for k, v in spec_test.items()})

    # Optional neural expert feeds into final gate, not just compared separately.
    neu_rows, neu_val, neu_test = train_optional_neural_expert(split_h, feat_h, horizon, seed + horizon * 401, settings)
    rows.extend(neu_rows)
    base_val.update({f"neural_{k}": v for k, v in neu_val.items()})
    base_test.update({f"neural_{k}": v for k, v in neu_test.items()})

    # Final gated hierarchy over all base/expert/formula outputs.
    gate_rows, gate_test = train_gated_meta_ensemble(split_h, base_val, base_test, feat_h, horizon, seed + horizon * 503, settings)
    rows.extend(gate_rows)
    base_test.update({f"gated_{k}": v for k, v in gate_test.items()})

    leaderboard = pd.DataFrame([r.as_dict() for r in rows]).sort_values(["return_mae", "directional_accuracy"], ascending=[True, False])
    pred_df = pd.DataFrame({"Date": split.test["Date"], "Close": split.test["Close"], f"target_return_{horizon}": split.test[f"target_return_{horizon}"]})
    for name, pred in base_test.items():
        if len(pred) == len(pred_df):
            pred_df[f"pred_{name}"] = pred
    return leaderboard, pred_df, base_test


def create_plots(out_dir: Path, final_leaderboard: pd.DataFrame, predictions: Dict[int, pd.DataFrame], data: pd.DataFrame) -> None:
    if final_leaderboard.empty:
        return
    # Overall MAE leaderboard.
    top = final_leaderboard.sort_values("return_mae").head(20).copy()
    top["label"] = top["track"] + " | " + top["model"] + " | h=" + top["horizon"].astype(str)
    plt.figure(figsize=(11, 8))
    plt.barh(top["label"][::-1], top["return_mae"][::-1])
    plt.xlabel("Test return MAE, lower is better")
    plt.title("Hierarchical Market Structure Lab: Test Return MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "01_overall_return_mae_leaderboard.png", dpi=180)
    plt.close()

    # Directional accuracy for rows with finite accuracy.
    d = final_leaderboard[np.isfinite(final_leaderboard["directional_accuracy"])].sort_values("directional_accuracy", ascending=False).head(20).copy()
    if not d.empty:
        d["label"] = d["track"] + " | " + d["model"] + " | h=" + d["horizon"].astype(str)
        plt.figure(figsize=(11, 8))
        plt.barh(d["label"][::-1], d["directional_accuracy"][::-1])
        plt.xlabel("Directional accuracy")
        plt.title("Directional Accuracy by Model")
        plt.tight_layout()
        plt.savefig(out_dir / "02_directional_accuracy_leaderboard.png", dpi=180)
        plt.close()

    # Active-event accuracy.
    a = final_leaderboard[np.isfinite(final_leaderboard["active_directional_accuracy"])].sort_values("active_directional_accuracy", ascending=False).head(20).copy()
    if not a.empty:
        a["label"] = a["track"] + " | " + a["model"] + " | h=" + a["horizon"].astype(str)
        plt.figure(figsize=(11, 8))
        plt.barh(a["label"][::-1], a["active_directional_accuracy"][::-1])
        plt.xlabel("Active/high-confidence directional accuracy")
        plt.title("Active-Set Directional Accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "03_active_directional_accuracy.png", dpi=180)
        plt.close()

    # Price forecast for best non-baseline/gated if possible.
    candidates = final_leaderboard[~final_leaderboard["track"].eq("baseline")].sort_values("return_mae")
    if not candidates.empty:
        best = candidates.iloc[0]
        h = int(best["horizon"])
        pred_df = predictions.get(h)
        if pred_df is not None:
            # Find matching pred column roughly.
            model = best["model"]
            possible = [c for c in pred_df.columns if model in c]
            if not possible:
                possible = [c for c in pred_df.columns if c.startswith("pred_gated_final_gated")]
            if not possible:
                possible = [c for c in pred_df.columns if c.startswith("pred_") and "zero_return" not in c]
            if possible:
                col = possible[0]
                true_price = pred_df["Close"] * np.exp(pred_df[f"target_return_{h}"])
                pred_price = pred_df["Close"] * np.exp(pred_df[col])
                plt.figure(figsize=(12, 5))
                plt.plot(pred_df["Date"], true_price, label="Actual future price")
                plt.plot(pred_df["Date"], pred_price, label="Predicted future price")
                plt.xlabel("Prediction date")
                plt.ylabel("BTC price")
                plt.title(f"Best Learned Model Forecast: {best['track']} / {best['model']} / h={h}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "04_best_model_price_forecast.png", dpi=180)
                plt.close()

    # Regime summary: average future return by hypercategory for best horizon.
    h0 = int(final_leaderboard.sort_values("return_mae").iloc[0]["horizon"])
    target = f"target_return_{h0}"
    if target in data.columns:
        hyper_cols = [c for c in data.columns if c.startswith("hyper_")][:40]
        rows = []
        for c in hyper_cols:
            m = data[c].fillna(0).to_numpy(dtype=float) > 0.5
            if m.sum() >= 30:
                rows.append((c, int(m.sum()), float(data.loc[m, target].mean()), float(data.loc[m, target].median())))
        if rows:
            reg = pd.DataFrame(rows, columns=["regime", "count", "mean_future_return", "median_future_return"]).sort_values("mean_future_return")
            reg.to_csv(out_dir / "regime_summary.csv", index=False)
            show = pd.concat([reg.head(10), reg.tail(10)])
            plt.figure(figsize=(12, 7))
            plt.barh(show["regime"], show["mean_future_return"])
            plt.xlabel(f"Mean future return, horizon={h0}")
            plt.title("Hypercategory Future Return Summary")
            plt.tight_layout()
            plt.savefig(out_dir / "05_regime_future_return_summary.png", dpi=180)
            plt.close()


def save_summary(out_dir: Path, config: LabConfig, settings: ModeSettings, leaderboard: pd.DataFrame, ablation: Optional[pd.DataFrame]) -> None:
    lines = []
    lines.append("# Hierarchical Market Structure Lab Summary")
    lines.append("")
    lines.append(f"Mode: `{config.mode}`")
    lines.append(f"Ticker: `{config.ticker}`")
    lines.append("")
    lines.append("## Architecture")
    lines.append("")
    lines.append("This run used a stacked hierarchy: multi-scale statistics -> regime/hypercategory flags -> formula/interaction features -> specialist experts -> final gated ensemble.")
    lines.append("")
    lines.append("## Best rows by test return MAE")
    lines.append("")
    if not leaderboard.empty:
        show_cols = ["track", "model", "horizon", "return_mae", "price_mae", "directional_accuracy", "active_coverage", "active_directional_accuracy"]
        lines.append(leaderboard[show_cols].head(15).to_markdown(index=False))
    lines.append("")
    if ablation is not None and not ablation.empty:
        lines.append("## Feature group ablation")
        lines.append("")
        lines.append(ablation.head(12).to_markdown(index=False))
        lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- The zero-return/persistence baseline is intentionally hard to beat.")
    lines.append("- The important test is whether learned models beat baselines on held-out chronological test data, not whether a price plot looks smooth.")
    lines.append("- Active/high-confidence accuracy should be treated cautiously if coverage is low.")
    lines.append("- Extremely high toy strategy returns are not reliable evidence by themselves; focus first on MAE, directional accuracy, and validation/test consistency.")
    (out_dir / "REPORT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(config: LabConfig) -> Path:
    set_seed(config.seed)
    settings = get_settings(config.mode)
    out_dir = ensure_dir(Path(config.output_root) / f"{safe_name(config.ticker)}_{config.mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    (out_dir / "run_config.json").write_text(pd.Series(asdict(config)).to_json(indent=2), encoding="utf-8")

    section("Hierarchical Crypto Market Structure Lab")
    print(f"Mode: {config.mode}")
    print(f"Ticker: {config.ticker}")
    print(f"Output directory: {out_dir}")
    print("Pipeline: raw data -> multi-scale stats -> regimes -> formulas -> specialists -> gated final prediction")

    section("Loading data")
    main, context = load_main_data(config)
    main.to_csv(out_dir / f"raw_{safe_name(config.ticker)}.csv", index=False)

    section("Feature engineering")
    data = engineer_features(main, context, settings, config.seed)
    data.to_csv(out_dir / "engineered_features.csv", index=False)

    section("Chronological split")
    split = chronological_split(data)
    print(f"train: rows={len(split.train):5d} | {split.train['Date'].min().date()} to {split.train['Date'].max().date()}")
    print(f"val  : rows={len(split.val):5d} | {split.val['Date'].min().date()} to {split.val['Date'].max().date()}")
    print(f"test : rows={len(split.test):5d} | {split.test['Date'].min().date()} to {split.test['Date'].max().date()}")
    feature_cols = numeric_feature_columns(data)
    # Drop columns that are entirely missing in training; they cannot help any model.
    feature_cols = [c for c in feature_cols if c in split.train.columns and split.train[c].notna().any()]
    print(f"Initial usable feature count: {len(feature_cols)}")

    split, feature_cols = add_latent_regimes(split, feature_cols, settings, config.seed)
    feature_cols = [c for c in feature_cols if c in split.train.columns and split.train[c].notna().any()]
    print(f"Feature count after latent regimes: {len(feature_cols)}")
    event_coverage = float(split.test.get("any_structure_event", pd.Series(0, index=split.test.index)).fillna(0).mean())
    print(f"Event/setup coverage in test: {event_coverage:.3f}")

    all_leaders = []
    predictions_by_h: Dict[int, pd.DataFrame] = {}
    pred_arrays_by_h: Dict[int, Dict[str, np.ndarray]] = {}
    for h in settings.horizons:
        leader, pred_df, pred_arrays = run_one_horizon(split, feature_cols, h, settings, config.seed, out_dir)
        leader.to_csv(out_dir / f"leaderboard_h{h}.csv", index=False)
        pred_df.to_csv(out_dir / f"predictions_h{h}.csv", index=False)
        all_leaders.append(leader)
        predictions_by_h[h] = pred_df
        pred_arrays_by_h[h] = pred_arrays

    final = pd.concat(all_leaders, ignore_index=True).sort_values(["return_mae", "directional_accuracy"], ascending=[True, False]) if all_leaders else pd.DataFrame()
    final.to_csv(out_dir / "final_leaderboard.csv", index=False)

    ablation = pd.DataFrame()
    if settings.include_ablation and settings.horizons:
        section("Feature group ablation")
        # Use the horizon of the best learned model if possible, else first horizon.
        h_ablate = int(final[~final["track"].eq("baseline")].sort_values("return_mae").iloc[0]["horizon"]) if not final[~final["track"].eq("baseline")].empty else settings.horizons[0]
        ablation = run_feature_ablation(split, feature_cols, h_ablate, config.seed + 999)
        if not ablation.empty:
            ablation.to_csv(out_dir / "feature_group_ablation.csv", index=False)
            print(ablation.head(10).to_string(index=False))

    create_plots(out_dir, final, predictions_by_h, data)
    save_summary(out_dir, config, settings, final, ablation)

    section("Best final test rows by return MAE")
    if not final.empty:
        show_cols = ["track", "model", "horizon", "return_mae", "price_mae", "directional_accuracy", "active_coverage", "active_directional_accuracy", "selected_by"]
        print(final[show_cols].head(20).to_string(index=False))

    section("Saved outputs")
    print(out_dir)
    print("Open these first:")
    for fname in [
        "REPORT_SUMMARY.md",
        "final_leaderboard.csv",
        "feature_group_ablation.csv",
        "01_overall_return_mae_leaderboard.png",
        "02_directional_accuracy_leaderboard.png",
        "03_active_directional_accuracy.png",
        "04_best_model_price_forecast.png",
        "05_regime_future_return_summary.png",
    ]:
        p = out_dir / fname
        if p.exists():
            print(f"  {p}")
    return out_dir
