from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, brier_score_loss

EPS = 1e-12


def direction_from_return(r) -> np.ndarray:
    return (np.asarray(r, dtype=float) > 0).astype(int)


def baseline_returns(df: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    z = np.zeros(len(df), dtype=float)
    out = {"zero": z}
    for w in [7, 14, 30, 60, 120]:
        c = f"ret_mean_{w}"
        if c in df.columns:
            out[f"ret_mean_{w}"] = df[c].to_numpy(dtype=float) * int(horizon)
    if f"prev_same_h_return_{horizon}" in df.columns:
        out["prev_same_h"] = df[f"prev_same_h_return_{horizon}"].fillna(0).to_numpy(dtype=float)
    elif f"prev_1d_return_scaled_to_h_{horizon}" in df.columns:
        out["prev_1d_scaled"] = df[f"prev_1d_return_scaled_to_h_{horizon}"].fillna(0).to_numpy(dtype=float)
    return out


def same_days_baseline_acc(df: pd.DataFrame, active: np.ndarray, horizon: int) -> dict[str, float]:
    y = df[f"target_direction_{horizon}"].to_numpy(dtype=int)
    out = {}
    for name, r in baseline_returns(df, horizon).items():
        pred = direction_from_return(r)
        out[f"same_active_{name}_acc"] = float(np.mean(pred[active] == y[active])) if active.sum() else np.nan
        out[f"same_active_{name}_mae"] = float(mean_absolute_error(df[f"target_return_{horizon}"].to_numpy(dtype=float)[active], r[active])) if active.sum() else np.nan
    return out


def evaluate_probability_model(df: pd.DataFrame, prob: np.ndarray, horizon: int, track: str, model: str, active_score=None, active_threshold=None, return_pred=None) -> dict:
    y_dir = df[f"target_direction_{horizon}"].to_numpy(dtype=int)
    y_ret = df[f"target_return_{horizon}"].to_numpy(dtype=float)
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, 1e-5, 1 - 1e-5)
    pred_dir = (prob >= 0.5).astype(int)
    if return_pred is None:
        # Direction probability converted to a small return estimate for MAE comparability.
        scale = float(np.nanmedian(np.abs(y_ret))) if len(y_ret) else 0.0
        return_pred = (prob - 0.5) * 2.0 * scale
    return_pred = np.asarray(return_pred, dtype=float)
    if active_score is None:
        active_score = np.abs(prob - 0.5)
    active_score = np.asarray(active_score, dtype=float)
    if active_threshold is None:
        active_threshold = -np.inf
    active = np.isfinite(active_score) & (active_score >= float(active_threshold))
    out = {
        "track": track,
        "model": model,
        "horizon": horizon,
        "n": len(df),
        "return_mae": float(mean_absolute_error(y_ret, return_pred)),
        "directional_accuracy": float(accuracy_score(y_dir, pred_dir)),
        "brier": float(brier_score_loss(y_dir, prob)),
        "active_n": int(active.sum()),
        "active_coverage": float(active.mean()),
        "active_directional_accuracy": float(np.mean(pred_dir[active] == y_dir[active])) if active.sum() else np.nan,
        "active_return_mae": float(mean_absolute_error(y_ret[active], return_pred[active])) if active.sum() else np.nan,
        "active_threshold": float(active_threshold) if np.isfinite(active_threshold) else np.nan,
    }
    out.update(same_days_baseline_acc(df, active, horizon))
    return out


def choose_confidence_threshold(val_df: pd.DataFrame, prob: np.ndarray, horizon: int, min_cov: float, max_cov: float = 0.70) -> dict:
    y = val_df[f"target_direction_{horizon}"].to_numpy(dtype=int)
    prob = np.asarray(prob, dtype=float)
    score = np.abs(prob - 0.5)
    pred = (prob >= 0.5).astype(int)
    baselines = baseline_returns(val_df, horizon)
    best = {"threshold": -np.inf, "edge": -1e9, "coverage": 1.0, "acc": float(np.mean(pred == y))}
    qs = np.linspace(0.10, 0.95, 18)
    for q in qs:
        thr = float(np.nanquantile(score, q))
        active = score >= thr
        cov = float(active.mean())
        if cov < min_cov or cov > max_cov or active.sum() < 20:
            continue
        acc = float(np.mean(pred[active] == y[active]))
        base_accs = []
        for r in baselines.values():
            base_accs.append(float(np.mean(direction_from_return(r)[active] == y[active])))
        edge = acc - max(base_accs)
        # prefer edge but mildly reward coverage.
        objective = edge + 0.02 * np.log(max(cov, 1e-6))
        if objective > best["edge"]:
            best = {"threshold": thr, "edge": objective, "raw_edge": edge, "coverage": cov, "acc": acc}
    return best
