from __future__ import annotations

import numpy as np
import pandas as pd

from .rules import EventRule
from ..evaluation.metrics import direction_from_return, baseline_returns

EPS = 1e-12


def _event_prior_predictions(train: pd.DataFrame, target_dir: str, target_ret: str, mask_train: np.ndarray):
    global_p = float(train[target_dir].mean())
    global_r = float(train[target_ret].mean())
    if mask_train.sum() < 2:
        return global_p, global_r
    p = float(train.loc[mask_train, target_dir].mean())
    r = float(train.loc[mask_train, target_ret].mean())
    # Shrink event prior toward global based on sample size.
    n = mask_train.sum()
    alpha = min(1.0, n / (n + 80.0))
    return alpha * p + (1 - alpha) * global_p, alpha * r + (1 - alpha) * global_r


def _score_on_split(eval_df: pd.DataFrame, mask: np.ndarray, horizon: int, p_up: float, mean_ret: float) -> dict:
    target_dir = f"target_direction_{horizon}"
    target_ret = f"target_return_{horizon}"
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "acc": np.nan, "edge": np.nan, "mae": np.nan, "best_base_acc": np.nan}
    y = eval_df[target_dir].to_numpy(dtype=int)
    yret = eval_df[target_ret].to_numpy(dtype=float)
    pred = np.full(len(eval_df), int(p_up >= 0.5), dtype=int)
    acc = float(np.mean(pred[mask] == y[mask]))
    base_accs = []
    for r in baseline_returns(eval_df, horizon).values():
        base_accs.append(float(np.mean(direction_from_return(r)[mask] == y[mask])))
    best_base = max(base_accs) if base_accs else np.nan
    mae = float(np.mean(np.abs(yret[mask] - mean_ret)))
    return {"n": n, "acc": acc, "edge": acc - best_base, "mae": mae, "best_base_acc": best_base}


def _walk_forward_score(df: pd.DataFrame, rule: EventRule, horizon: int, config) -> dict:
    n = len(df)
    folds = int(config.walk_forward_folds)
    if n < 400 or folds < 2:
        return {"wf_edge_mean": np.nan, "wf_positive_frac": 0.0, "wf_folds": 0}
    start = int(n * 0.45)
    fold_size = max(40, (n - start) // folds)
    edges = []
    for k in range(folds):
        val_start = start + k * fold_size
        val_end = min(n, val_start + fold_size)
        tr_end = max(0, val_start - int(config.walk_forward_gap))
        if val_end - val_start < 30 or tr_end < 200:
            continue
        tr = df.iloc[:tr_end]
        va = df.iloc[val_start:val_end]
        mt = rule.eval(tr)
        mv = rule.eval(va)
        if mt.sum() < config.min_train_event_n // 2 or mv.sum() < max(10, config.min_val_event_n // 2):
            continue
        p, r = _event_prior_predictions(tr, f"target_direction_{horizon}", f"target_return_{horizon}", mt)
        sc = _score_on_split(va, mv, horizon, p, r)
        if np.isfinite(sc["edge"]):
            edges.append(sc["edge"])
    if not edges:
        return {"wf_edge_mean": np.nan, "wf_positive_frac": 0.0, "wf_folds": 0}
    edges = np.array(edges, dtype=float)
    return {"wf_edge_mean": float(np.mean(edges)), "wf_positive_frac": float(np.mean(edges > 0)), "wf_folds": int(len(edges))}


def _perturbation_score(train, val, rule, horizon, config, rng) -> dict:
    target_dir = f"target_direction_{horizon}"
    target_ret = f"target_return_{horizon}"
    mt = rule.eval(train)
    p, r = _event_prior_predictions(train, target_dir, target_ret, mt)
    wins = 0; valid = 0; edges = []
    for _ in range(int(config.perturb_trials)):
        mv = rule.eval(val, rng=rng, noise_scale=config.perturb_noise_scale, threshold_scale=config.perturb_threshold_scale)
        if mv.sum() < config.min_val_event_n:
            continue
        sc = _score_on_split(val, mv, horizon, p, r)
        if np.isfinite(sc["edge"]):
            valid += 1
            edges.append(sc["edge"])
            wins += int(sc["edge"] > 0)
    if valid == 0:
        return {"noise_survival": 0.0, "noise_edge_mean": np.nan, "noise_trials_valid": 0}
    return {"noise_survival": float(wins / valid), "noise_edge_mean": float(np.mean(edges)), "noise_trials_valid": valid}


def score_candidate_rules(train: pd.DataFrame, val: pd.DataFrame, rules: list[EventRule], horizon: int, config) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + horizon * 100)
    rows = []
    pretest = pd.concat([train, val], axis=0).copy()
    target_dir = f"target_direction_{horizon}"
    target_ret = f"target_return_{horizon}"
    for i, rule in enumerate(rules):
        try:
            mt = rule.eval(train)
            mv = rule.eval(val)
            if mt.sum() < config.min_train_event_n or mv.sum() < config.min_val_event_n:
                continue
            cov = float(mv.mean())
            if cov < config.min_active_coverage or cov > config.max_active_coverage:
                continue
            p, r = _event_prior_predictions(train, target_dir, target_ret, mt)
            val_sc = _score_on_split(val, mv, horizon, p, r)
            pert = _perturbation_score(train, val, rule, horizon, config, rng)
            wf = _walk_forward_score(pretest, rule, horizon, config)
            sample_score = min(1.0, mv.sum() / 120.0)
            edge = val_sc["edge"] if np.isfinite(val_sc["edge"]) else -1.0
            wf_pos = wf["wf_positive_frac"]
            noise_surv = pert["noise_survival"]
            robust_score = 0.40 * edge + 0.25 * (noise_surv - 0.5) + 0.25 * (wf_pos - 0.5) + 0.10 * sample_score
            rows.append({
                "rule_id": i, "rule_name": rule.name, "source": rule.source, "horizon": horizon,
                "train_n": int(mt.sum()), "val_n": int(mv.sum()), "val_coverage": cov,
                "event_train_p_up": p, "event_train_mean_ret": r,
                "val_acc": val_sc["acc"], "val_best_base_acc": val_sc["best_base_acc"], "val_edge": edge, "val_mae": val_sc["mae"],
                **pert, **wf,
                "robust_score": robust_score,
            })
        except Exception as e:
            # Skip brittle rules; they did not earn the pattern.
            continue
    return pd.DataFrame(rows).sort_values("robust_score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
