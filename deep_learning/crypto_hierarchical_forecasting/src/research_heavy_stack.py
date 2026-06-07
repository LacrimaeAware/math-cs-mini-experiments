"""
Research-heavy stacked market-structure experiment.

This is intentionally more expensive than RUN_HEAVY.py. It does repeated
expanding-window base-model training to create out-of-fold expert predictions,
then trains a meta-model/gate on those predictions. This is closer to the user's
intended hierarchy:

multi-scale features -> regimes/formulas -> experts -> out-of-fold gate -> final prediction

The goal is still honest testing against persistence, not optimistic in-sample blending.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .hierarchical_market_structure import (
    DataSplit,
    LabConfig,
    MetricRow,
    add_latent_regimes,
    baseline_predictions,
    chronological_split,
    choose_active_threshold,
    engineer_features,
    ensure_dir,
    evaluate_predictions,
    feature_groups,
    generate_formula_layer,
    get_settings,
    load_main_data,
    numeric_feature_columns,
    safe_name,
    section,
    set_seed,
    top_correlated_features,
)


@dataclass
class ResearchConfig:
    mode: str = "research_medium"
    ticker: str = "BTC-USD"
    start: str = "2016-01-01"
    end: Optional[str] = None
    output_root: str = "hierarchical_market_outputs"
    horizons: Tuple[int, ...] = (1, 2, 3)
    n_folds: int = 4
    max_base_features: int = 220
    include_specialists: bool = True
    include_big_tree: bool = False
    seed: int = 42
    local_csv: Optional[str] = None
    use_synthetic: bool = False


def _model_specs(seed: int, include_big_tree: bool) -> Dict[str, Pipeline]:
    specs: Dict[str, Pipeline] = {
        "ridge_stack": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", Ridge(alpha=20.0)),
        ]),
        "elastic_stack": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", ElasticNet(alpha=0.0007, l1_ratio=0.25, max_iter=8000, random_state=seed)),
        ]),
        "extra_trees_400": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=8,
                max_features=0.40,
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "hist_gb_250": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.025,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=18,
                l2_regularization=0.05,
                random_state=seed,
            )),
        ]),
    }
    if include_big_tree:
        specs["extra_trees_900_wide"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=900,
                max_depth=None,
                min_samples_leaf=6,
                max_features=0.55,
                random_state=seed + 91,
                n_jobs=-1,
            )),
        ])
        specs["random_forest_500"] = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=500,
                max_depth=14,
                min_samples_leaf=8,
                max_features=0.45,
                random_state=seed + 103,
                n_jobs=-1,
            )),
        ])
    return specs


def _direction_model(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("model", LogisticRegression(C=0.4, penalty="l2", class_weight="balanced", max_iter=2000, random_state=seed)),
    ])


def _meta_specs(seed: int) -> Dict[str, Pipeline]:
    return {
        "meta_ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ]),
        "meta_huber": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", HuberRegressor(alpha=0.0005, epsilon=1.35, max_iter=1000)),
        ]),
        "meta_elastic": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", ElasticNet(alpha=0.0003, l1_ratio=0.35, max_iter=8000, random_state=seed)),
        ]),
    }


def _clean_X(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    return df[list(cols)].replace([np.inf, -np.inf], np.nan)


def _select_features(train: pd.DataFrame, feature_cols: List[str], horizon: int, max_features: int) -> List[str]:
    target = f"target_return_{horizon}"
    # Keep a structured mixture: correlated features + formula/category features even if weak.
    correlated = top_correlated_features(train, feature_cols, target, max_features)
    must_keep = [
        c for c in feature_cols
        if c.startswith("flag_") or c.startswith("hyper_") or c.startswith("cross_") or c.startswith("latent_regime_") or c.startswith("form_")
    ]
    out: List[str] = []
    for c in correlated + must_keep:
        if c not in out:
            out.append(c)
        if len(out) >= max_features + 120:
            break
    return out


def _fold_ranges(n: int, n_folds: int, min_train: int) -> List[Tuple[int, int, int]]:
    if n <= min_train + 80:
        return []
    remaining = n - min_train
    fold_size = max(60, remaining // n_folds)
    folds: List[Tuple[int, int, int]] = []
    start = min_train
    for _ in range(n_folds):
        end = min(n, start + fold_size)
        if end - start < 40:
            break
        folds.append((0, start, end))
        start = end
        if n - start < 40:
            break
    return folds


def _fit_base_predict(
    train: pd.DataFrame,
    pred_df: pd.DataFrame,
    cols: List[str],
    horizon: int,
    seed: int,
    include_big_tree: bool,
    include_specialists: bool,
) -> Dict[str, np.ndarray]:
    y = train[f"target_return_{horizon}"].to_numpy(dtype=float)
    preds: Dict[str, np.ndarray] = {}

    # Baselines are experts too.
    for name, p in baseline_predictions(pred_df, horizon).items():
        preds[f"baseline_{name}"] = np.asarray(p, dtype=float)

    Xtr = _clean_X(train, cols)
    Xp = _clean_X(pred_df, cols)

    for i, (name, model) in enumerate(_model_specs(seed, include_big_tree).items()):
        try:
            m = model
            m.fit(Xtr, y)
            preds[f"global_{name}"] = m.predict(Xp)
        except Exception as exc:
            print(f"    skipped {name}: {exc}")

    # Direction model converted to return by using train median absolute move.
    try:
        d = (y > 0).astype(int)
        if len(np.unique(d)) == 2:
            clf = _direction_model(seed + 17)
            clf.fit(Xtr, d)
            prob = clf.predict_proba(Xp)[:, 1]
            mag = float(np.nanmedian(np.abs(y)))
            preds["dir_logistic_prob_x_mag"] = (prob - 0.5) * 2.0 * mag
    except Exception as exc:
        print(f"    skipped direction model: {exc}")

    # Specialist experts: train only inside categories; predict only when category active.
    if include_specialists:
        flag_cols = [c for c in train.columns if c.startswith("flag_") or c.startswith("hyper_") or c.startswith("cross_")]
        candidates = []
        for c in flag_cols:
            tr_active = train[c].fillna(0).to_numpy(dtype=float) > 0.5
            pr_active = pred_df[c].fillna(0).to_numpy(dtype=float) > 0.5 if c in pred_df.columns else np.zeros(len(pred_df), dtype=bool)
            if tr_active.sum() >= 160 and pr_active.sum() >= 8:
                # rank by difference in future return conditional on category
                diff = abs(float(np.nanmean(y[tr_active]) - np.nanmean(y[~tr_active]))) if (~tr_active).sum() > 10 else 0.0
                candidates.append((diff, c, tr_active, pr_active))
        candidates.sort(reverse=True)
        for j, (_, flag, tr_active, pr_active) in enumerate(candidates[:10]):
            try:
                spec = ExtraTreesRegressor(
                    n_estimators=260,
                    max_depth=10,
                    min_samples_leaf=6,
                    max_features=0.45,
                    random_state=seed + 500 + j,
                    n_jobs=-1,
                )
                spec.fit(Xtr.loc[tr_active], y[tr_active])
                out = np.zeros(len(pred_df), dtype=float)
                out[pr_active] = spec.predict(Xp.loc[pr_active])
                preds[f"specialist_{flag[:80]}"] = out
            except Exception:
                continue

    return preds


def _evaluate_prediction_frame(df: pd.DataFrame, pred_map: Dict[str, np.ndarray], horizon: int, prefix: str) -> pd.DataFrame:
    rows = []
    for name, pred in pred_map.items():
        try:
            row = evaluate_predictions(df, pred, horizon, prefix, name, "test", active_mask=np.abs(pred) > 0).as_dict()
            rows.append(row)
        except Exception:
            pass
    return pd.DataFrame(rows)


def _train_meta_stack(
    oof_df: pd.DataFrame,
    test_expert_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    seed: int,
    min_active_coverage: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
    pred_cols = [c for c in oof_df.columns if c.startswith("expert__")]
    y = oof_df[f"target_return_{horizon}"].to_numpy(dtype=float)
    valid = np.isfinite(y)
    for c in pred_cols:
        valid &= np.isfinite(oof_df[c].to_numpy(dtype=float))
    oof = oof_df.loc[valid].copy().reset_index(drop=True)
    if len(oof) < 200 or not pred_cols:
        return pd.DataFrame(), {}, "not enough OOF rows"

    # Chronological meta train/validation split. This prevents tuning alpha/threshold on the same rows used to fit meta.
    cut = int(len(oof) * 0.72)
    meta_train = oof.iloc[:cut].copy()
    meta_val = oof.iloc[cut:].copy()
    Xtr = meta_train[pred_cols]
    ytr = meta_train[f"target_return_{horizon}"].to_numpy(dtype=float)
    Xv = meta_val[pred_cols]
    yv = meta_val[f"target_return_{horizon}"].to_numpy(dtype=float)
    Xt = test_expert_df[pred_cols]

    rows = []
    out_preds: Dict[str, np.ndarray] = {}
    notes = []
    for name, model in _meta_specs(seed).items():
        try:
            model.fit(Xtr, ytr)
            val_raw = model.predict(Xv)
            test_raw = model.predict(Xt)
        except Exception as exc:
            notes.append(f"{name} failed: {exc}")
            continue

        # Validation-chosen shrinkage and active threshold.
        best = (float("inf"), 0.0, val_raw)
        for alpha in np.linspace(0.0, 1.0, 41):
            p = alpha * val_raw
            mae = mean_absolute_error(yv, p)
            if mae < best[0]:
                best = (float(mae), float(alpha), p)
        _, alpha, val_pred = best
        test_pred = alpha * test_raw
        thr = choose_active_threshold(yv, val_pred, min_active_coverage)
        row = evaluate_predictions(
            test_df,
            test_pred,
            horizon,
            "true_oof_stacked_meta",
            name,
            "test",
            selected_by=f"OOF_meta;alpha={alpha:.3f};thr={thr:.6f};n_oof={len(oof)};n_experts={len(pred_cols)}",
            active_mask=np.abs(test_pred) >= thr,
        ).as_dict()
        rows.append(row)
        out_preds[f"meta_{name}"] = test_pred

    # Also include a validation-weighted expert blend using only OOF expert validation scores.
    try:
        scored = []
        for c in pred_cols:
            mae = mean_absolute_error(yv, Xv[c].to_numpy(dtype=float))
            if np.isfinite(mae):
                scored.append((mae, c))
        scored.sort()
        top = scored[: min(12, len(scored))]
        w = np.array([1.0 / ((m + 1e-6) ** 2) for m, _ in top], dtype=float)
        w = w / (w.sum() + 1e-12)
        val_blend = np.zeros(len(meta_val), dtype=float)
        test_blend = np.zeros(len(test_df), dtype=float)
        for wi, (_, c) in zip(w, top):
            val_blend += wi * Xv[c].to_numpy(dtype=float)
            test_blend += wi * Xt[c].to_numpy(dtype=float)
        best = (float("inf"), 0.0)
        for alpha in np.linspace(0.0, 1.0, 41):
            mae = mean_absolute_error(yv, alpha * val_blend)
            if mae < best[0]:
                best = (float(mae), float(alpha))
        alpha = best[1]
        test_pred = alpha * test_blend
        val_pred = alpha * val_blend
        thr = choose_active_threshold(yv, val_pred, min_active_coverage)
        expert_names = ",".join([c.replace("expert__", "") for _, c in top])
        row = evaluate_predictions(
            test_df,
            test_pred,
            horizon,
            "true_oof_stacked_meta",
            "weighted_oof_expert_blend",
            "test",
            selected_by=f"OOF_weighted;alpha={alpha:.3f};thr={thr:.6f};experts={expert_names}",
            active_mask=np.abs(test_pred) >= thr,
        ).as_dict()
        rows.append(row)
        out_preds["meta_weighted_oof_expert_blend"] = test_pred
    except Exception as exc:
        notes.append(f"weighted blend failed: {exc}")

    return pd.DataFrame(rows), out_preds, "; ".join(notes)


def _plot_leaderboard(out_dir: Path, leaderboard: pd.DataFrame) -> None:
    if leaderboard.empty:
        return
    top = leaderboard.sort_values("return_mae").head(25).copy()
    top["label"] = top["track"] + " | " + top["model"] + " | h=" + top["horizon"].astype(str)
    plt.figure(figsize=(12, 9))
    plt.barh(top["label"][::-1], top["return_mae"][::-1])
    plt.xlabel("Test return MAE, lower is better")
    plt.title("Research-heavy true OOF stack: test return MAE")
    plt.tight_layout()
    plt.savefig(out_dir / "01_research_heavy_return_mae.png", dpi=180)
    plt.close()

    topd = leaderboard.dropna(subset=["active_directional_accuracy"]).sort_values("active_directional_accuracy", ascending=False).head(25).copy()
    if len(topd):
        topd["label"] = topd["track"] + " | " + topd["model"] + " | h=" + topd["horizon"].astype(str)
        plt.figure(figsize=(12, 9))
        plt.barh(topd["label"][::-1], topd["active_directional_accuracy"][::-1])
        plt.xlabel("Active directional accuracy")
        plt.title("Research-heavy true OOF stack: active/setup directional accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "02_research_heavy_active_direction.png", dpi=180)
        plt.close()


def _write_summary(out_dir: Path, cfg: ResearchConfig, leaderboard: pd.DataFrame, fold_log: List[dict]) -> None:
    lines: List[str] = []
    lines.append("# Research-heavy hierarchical stack summary")
    lines.append("")
    lines.append("This run is intentionally different from the earlier fast heavy mode. It repeatedly retrains base experts over expanding walk-forward folds to create out-of-fold expert predictions, then trains a meta-gate on those OOF predictions.")
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(asdict(cfg), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Best rows by test return MAE")
    if leaderboard.empty:
        lines.append("No leaderboard rows were produced.")
    else:
        cols = ["track", "model", "horizon", "return_mae", "price_mae", "directional_accuracy", "active_coverage", "active_directional_accuracy", "selected_by"]
        lines.append(leaderboard[cols].sort_values("return_mae").head(20).to_string(index=False))
    lines.append("")
    lines.append("## Fold log")
    lines.append(pd.DataFrame(fold_log).to_string(index=False) if fold_log else "No fold log.")
    (out_dir / "RESEARCH_HEAVY_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def run_research_heavy(cfg: ResearchConfig) -> Path:
    start_time = time.time()
    set_seed(cfg.seed)
    section("Research-heavy hierarchical OOF stacking experiment")
    print(f"Mode: {cfg.mode}")
    print("This version is meant to run longer: expanding walk-forward base experts -> OOF meta-gate -> final test.")

    # Use the original heavy feature mode for all research modes so features are rich.
    settings = get_settings("heavy")
    settings = settings.__class__(
        horizons=cfg.horizons,
        scales=settings.scales,
        spectral_windows=settings.spectral_windows,
        n_random_shape_kernels=settings.n_random_shape_kernels,
        n_formula_base_features=settings.n_formula_base_features,
        n_formula_category_features=settings.n_formula_category_features,
        max_selected_formulas=settings.max_selected_formulas,
        n_latent_regimes=settings.n_latent_regimes,
        min_specialist_train=settings.min_specialist_train,
        min_event_val=settings.min_event_val,
        min_active_coverage=0.08,
        include_neural=False,
        neural_epochs=0,
        neural_patience=0,
        include_walk_forward=True,
        include_ablation=False,
    )

    out_dir = ensure_dir(Path(cfg.output_root) / f"{safe_name(cfg.ticker)}_{cfg.mode}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    (out_dir / "research_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    lab_cfg = LabConfig(
        mode="heavy",
        ticker=cfg.ticker,
        start=cfg.start,
        end=cfg.end,
        output_root=cfg.output_root,
        seed=cfg.seed,
        local_csv=cfg.local_csv,
        use_synthetic=cfg.use_synthetic,
    )
    section("Loading and engineering features")
    main, context = load_main_data(lab_cfg)
    data = engineer_features(main, context, settings, cfg.seed)
    split = chronological_split(data, train_frac=0.70, val_frac=0.15)
    pretest = pd.concat([split.train, split.val], axis=0).reset_index(drop=True)
    test = split.test.copy().reset_index(drop=True)
    base_cols = numeric_feature_columns(data)
    ds = DataSplit(train=split.train, val=split.val, test=split.test)
    ds, base_cols = add_latent_regimes(ds, base_cols, settings, cfg.seed + 19)
    pretest = pd.concat([ds.train, ds.val], axis=0).reset_index(drop=True)
    test = ds.test.copy().reset_index(drop=True)
    base_cols = numeric_feature_columns(pretest)
    print(f"Pretest rows: {len(pretest)} | final test rows: {len(test)} | base feature count: {len(base_cols)}")

    all_rows: List[pd.DataFrame] = []
    all_preds: Dict[str, pd.DataFrame] = {}
    fold_log: List[dict] = []

    for horizon in cfg.horizons:
        section(f"Research horizon {horizon} days")
        # Formula features selected only with first 70% train vs calibration; final test is untouched.
        split_for_formula = DataSplit(train=ds.train.copy(), val=ds.val.copy(), test=ds.test.copy())
        split_f, feature_cols, formulas = generate_formula_layer(split_for_formula, base_cols, horizon, settings, cfg.seed + horizon * 37)
        formula_path = out_dir / f"research_selected_formulas_h{horizon}.txt"
        formula_path.write_text("\n".join(formulas[:500]), encoding="utf-8")
        pre = pd.concat([split_f.train, split_f.val], axis=0).reset_index(drop=True)
        te = split_f.test.copy().reset_index(drop=True)

        selected = _select_features(split_f.train, feature_cols, horizon, cfg.max_base_features)
        print(f"Selected features for h={horizon}: {len(selected)} including {len([c for c in selected if c.startswith('form_')])} formulas")

        n = len(pre)
        min_train = max(950, int(n * 0.45))
        folds = _fold_ranges(n, cfg.n_folds, min_train)
        if not folds:
            print("Not enough rows for OOF folds; skipping horizon.")
            continue

        # Start OOF table with targets and dates. Expert columns are filled fold by fold.
        oof = pre[["Date", "Close", f"target_return_{horizon}"]].copy()
        expert_names_seen: List[str] = []
        final_expert_predictions: Dict[str, np.ndarray] = {}

        # Generate OOF predictions.
        for fold_id, (tr0, tr1, va1) in enumerate(folds, start=1):
            train_fold = pre.iloc[tr0:tr1].copy().reset_index(drop=True)
            val_fold = pre.iloc[tr1:va1].copy().reset_index(drop=True)
            print(f"  fold {fold_id}/{len(folds)}: train rows={len(train_fold)} | oof rows={len(val_fold)} | {val_fold['Date'].min().date()} to {val_fold['Date'].max().date()}")
            preds_fold = _fit_base_predict(
                train_fold,
                val_fold,
                selected,
                horizon,
                cfg.seed + horizon * 1000 + fold_id * 11,
                cfg.include_big_tree,
                cfg.include_specialists,
            )
            for name, pred in preds_fold.items():
                col = f"expert__{name}"
                if col not in oof.columns:
                    oof[col] = np.nan
                    expert_names_seen.append(name)
                oof.loc[tr1:va1 - 1, col] = pred[: len(val_fold)]
            fold_log.append({"horizon": horizon, "fold": fold_id, "train_rows": len(train_fold), "oof_rows": len(val_fold), "n_experts": len(preds_fold)})

        # Train final base experts on all pretest and predict final test.
        print("  training final base experts on all pretest rows for final test prediction...")
        final_expert_predictions = _fit_base_predict(
            pre,
            te,
            selected,
            horizon,
            cfg.seed + horizon * 2000,
            cfg.include_big_tree,
            cfg.include_specialists,
        )
        test_expert_df = pd.DataFrame({f"expert__{k}": v for k, v in final_expert_predictions.items()})
        # Keep expert columns that have both OOF and test predictions.
        common_cols = [c for c in oof.columns if c.startswith("expert__") and c in test_expert_df.columns]
        oof = oof[["Date", "Close", f"target_return_{horizon}"] + common_cols]
        test_expert_df = test_expert_df[common_cols]
        print(f"  common experts for meta-gate: {len(common_cols)}")

        # Evaluate final base experts directly on test for comparison.
        base_eval = _evaluate_prediction_frame(te, final_expert_predictions, horizon, "final_base_expert")
        if not base_eval.empty:
            all_rows.append(base_eval)

        meta_eval, meta_preds, notes = _train_meta_stack(oof, test_expert_df, te, horizon, cfg.seed + horizon * 3000, 0.08)
        if notes:
            print(f"  meta notes: {notes}")
        if not meta_eval.empty:
            all_rows.append(meta_eval)

        pred_out = te[["Date", "Close", f"target_return_{horizon}"]].copy()
        for name, pred in final_expert_predictions.items():
            pred_out[f"pred_base_{name}"] = pred
        for name, pred in meta_preds.items():
            pred_out[f"pred_{name}"] = pred
        pred_out.to_csv(out_dir / f"research_predictions_h{horizon}.csv", index=False)
        all_preds[f"h{horizon}"] = pred_out

    if all_rows:
        leaderboard = pd.concat(all_rows, axis=0, ignore_index=True)
        leaderboard = leaderboard.sort_values(["return_mae", "directional_accuracy"], ascending=[True, False])
    else:
        leaderboard = pd.DataFrame()
    leaderboard.to_csv(out_dir / "research_heavy_leaderboard.csv", index=False)
    pd.DataFrame(fold_log).to_csv(out_dir / "research_fold_log.csv", index=False)

    _plot_leaderboard(out_dir, leaderboard)
    _write_summary(out_dir, cfg, leaderboard, fold_log)

    elapsed = time.time() - start_time
    section("Research-heavy run complete")
    print(f"Output directory: {out_dir}")
    print(f"Elapsed seconds: {elapsed:.1f}")
    print("Open these first:")
    print(f"  {out_dir / 'RESEARCH_HEAVY_SUMMARY.md'}")
    print(f"  {out_dir / 'research_heavy_leaderboard.csv'}")
    print(f"  {out_dir / 'research_fold_log.csv'}")
    print(f"  {out_dir / '01_research_heavy_return_mae.png'}")
    print(f"  {out_dir / '02_research_heavy_active_direction.png'}")
    return out_dir
