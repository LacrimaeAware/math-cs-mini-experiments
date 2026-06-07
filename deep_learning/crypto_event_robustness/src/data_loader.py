from __future__ import annotations

import numpy as np
import pandas as pd


def _flatten_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def load_market_data(config) -> pd.DataFrame:
    if config.use_synthetic:
        return make_synthetic_market(seed=config.seed)

    if config.local_csv:
        raw = pd.read_csv(config.local_csv)
        if "Date" in raw.columns:
            raw["Date"] = pd.to_datetime(raw["Date"])
            raw = raw.set_index("Date")
        raw.index = pd.to_datetime(raw.index)
        return standardize_ohlcv(raw)

    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("yfinance is required for live ticker downloads. Install requirements.txt or use synthetic mode.") from e

    print(f"Main ticker: {config.ticker} | start={config.start} | end={config.end}")
    main = yf.download(config.ticker, start=config.start, end=config.end, interval="1d", auto_adjust=False, progress=False)
    main = _flatten_yfinance(main)
    main = standardize_ohlcv(main)

    for ticker in config.context_tickers:
        try:
            print(f"Downloading context ticker: {ticker}")
            ctx = yf.download(ticker, start=config.start, end=config.end, interval="1d", auto_adjust=False, progress=False)
            ctx = _flatten_yfinance(ctx)
            ctx = standardize_ohlcv(ctx)
            name = ticker.replace("-", "_").replace("^", "").replace(".", "_")
            main[f"ctx_{name}_close"] = ctx["Close"]
            main[f"ctx_{name}_ret1"] = np.log(ctx["Close"]).diff()
            main[f"ctx_{name}_ret7"] = np.log(ctx["Close"]).diff(7)
            main[f"ctx_{name}_vol14"] = np.log(ctx["Close"]).diff().rolling(14).std()
        except Exception as e:
            print(f"  WARNING: failed context ticker {ticker}: {e}")

    main = main.sort_index()
    main = main.ffill()
    main = main.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return main


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    cols = {c.lower(): c for c in out.columns}
    wanted = {}
    for target in ["Open", "High", "Low", "Close", "Volume"]:
        if target in out.columns:
            wanted[target] = target
        elif target.lower() in cols:
            wanted[cols[target.lower()]] = target
    out = out.rename(columns=wanted)
    missing = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c not in out.columns]
    if missing:
        raise ValueError(f"OHLCV columns missing: {missing}")
    return out[[c for c in out.columns if c in ["Open", "High", "Low", "Close", "Volume"] or c.startswith("ctx_")]].copy()


def make_synthetic_market(seed: int = 42, n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-01", periods=n, freq="D")
    # Synthetic returns with weak events: dip-after-uptrend has slight bounce bias; Tuesdays tiny noise.
    vol = 0.018 + 0.012 * (np.sin(np.arange(n) / 75.0) > 0).astype(float)
    drift = 0.00025 + 0.0002 * np.sin(np.arange(n) / 180.0)
    ret = drift + rng.normal(0, vol)
    for i in range(60, n - 5):
        recent = ret[i-5:i].sum()
        trend = ret[i-60:i].sum()
        if recent < -0.06 and trend > 0:
            ret[i+1] += 0.012
            ret[i+2] += 0.006
        if recent > 0.08 and trend < 0:
            ret[i+1] -= 0.010
    close = 10000 * np.exp(np.cumsum(ret))
    open_ = close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, n))
    volume = np.exp(18 + rng.normal(0, 0.5, n))
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=dates)
    return df
