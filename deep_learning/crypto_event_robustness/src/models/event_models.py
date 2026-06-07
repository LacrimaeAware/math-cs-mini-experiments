from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

from ..evaluation.metrics import evaluate_probability_model, choose_confidence_threshold


def select_model_features(
    train: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    max_features: int,
    seed: int,
) -> list[str]:
    """
    Fast, conservative feature selection by absolute correlation to the direction target.

    Mutual information can be slow/noisy on small active-event subsets. Correlation is simpler,
    more stable, and sufficient for selecting a compact candidate set.
    """
    good = []
    y = train[target].astype(float)

    for c in feature_cols:
        if c not in train.columns or not pd.api.types.is_numeric_dtype(train[c]):
            continue

        s = train[c].replace([np.inf, -np.inf], np.nan)

        if s.notna().sum() < 50:
            continue

        if s.nunique(dropna=True) <= 1:
            continue

        good.append(c)

    if len(good) <= max_features:
        return good

    scores = []

    for c in good:
        x = train[c].replace([np.inf, -np.inf], np.nan).fillna(train[c].median())

        try:
            corr = np.corrcoef(x.to_numpy(dtype=float), y.to_numpy(dtype=float))[0, 1]
            if not np.isfinite(corr):
                corr = 0.0
        except Exception:
            corr = 0.0

        # Slight preference for selected event flags and learned/manual event variables.
        bonus = 0.02 if (
            c.startswith("selected_event_")
            or c.startswith("seed_")
            or c.startswith("learned_")
        ) else 0.0

        scores.append((abs(corr) + bonus, c))

    scores.sort(reverse=True)
    return [c for _, c in scores[:max_features]]


def _hist_model(seed: int):
    # Lightweight tree model used as a stable stand-in for histogram boosting.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=180,
            max_depth=5,
            min_samples_leaf=25,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def _extra_model(seed: int):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=30,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def _logistic_model(seed: int):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            C=0.5,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])


def add_selected_event_flags(
    df: pd.DataFrame,
    selected_rules,
    prefix: str = "selected_event",
) -> pd.DataFrame:
    out = df.copy()
    frames = []

    for i, rule in enumerate(selected_rules):
        frames.append(
            pd.Series(
                rule.eval(out).astype(float),
                index=out.index,
                name=f"{prefix}_{i:02d}",
            )
        )

    if frames:
        out = pd.concat([out, pd.concat(frames, axis=1)], axis=1)

    return out


def prior_event_ensemble(train, val, test, selected_rules, selected_scores, horizon, cost_bps=10.0):
    rows = []
    pred_payload = {}

    if not selected_rules:
        return rows, pred_payload

    weights = np.array([max(float(s), 0.0) + 1e-6 for s in selected_scores], dtype=float)
    weights = weights / weights.sum()

    target_dir = f"target_direction_{horizon}"
    target_ret = f"target_return_{horizon}"

    p_val = np.zeros(len(val))
    p_test = np.zeros(len(test))
    r_val = np.zeros(len(val))
    r_test = np.zeros(len(test))

    active_val = np.zeros(len(val), dtype=bool)
    active_test = np.zeros(len(test), dtype=bool)

    score_val = np.zeros(len(val))
    score_test = np.zeros(len(test))

    for w, rule in zip(weights, selected_rules):
        mt = rule.eval(train)
        mv = rule.eval(val)
        ms = rule.eval(test)

        if mt.sum() < 10:
            continue

        p = float(train.loc[mt, target_dir].mean())
        r = float(train.loc[mt, target_ret].mean())

        p_val[mv] += w * p
        p_test[ms] += w * p
        r_val[mv] += w * r
        r_test[ms] += w * r

        score_val[mv] += w * abs(p - 0.5)
        score_test[ms] += w * abs(p - 0.5)

        active_val |= mv
        active_test |= ms

    # If no selected event active, probability remains neutral.
    p_val = np.where(active_val, p_val, 0.5)
    p_test = np.where(active_test, p_test, 0.5)

    score_val = np.where(active_val, score_val, 0.0)
    score_test = np.where(active_test, score_test, 0.0)

    thr_info = choose_confidence_threshold(
        val,
        p_val,
        horizon,
        min_cov=0.03,
        max_cov=0.70,
    )

    row = evaluate_probability_model(
        test,
        p_test,
        horizon,
        "robust_event_prior",
        "weighted_event_priors",
        active_score=score_test,
        active_threshold=thr_info["threshold"],
        return_pred=r_test,
    )

    row["selected_by"] = (
        f"robust_event_priors;"
        f"n_events={len(selected_rules)};"
        f"thr={thr_info['threshold']:.6f}"
    )

    rows.append(row)

    pred_payload[f"robust_event_prior_h{horizon}"] = {
        "prob": p_test,
        "return_pred": r_test,
        "score": score_test,
        "threshold": thr_info["threshold"],
        "active": score_test >= thr_info["threshold"],
    }

    return rows, pred_payload


def _row_get(row: dict, key: str, default=np.nan):
    try:
        return row.get(key, default)
    except Exception:
        return default


def _active_edge_vs_best_baseline(row: dict) -> float:
    acc = _row_get(row, "active_directional_accuracy", np.nan)

    if not np.isfinite(acc):
        return -1e9

    base_accs = []

    for k, v in row.items():
        if k.startswith("same_active_") and k.endswith("_acc"):
            try:
                v = float(v)
                if np.isfinite(v):
                    base_accs.append(v)
            except Exception:
                pass

    if not base_accs:
        return -1e9

    return float(acc - max(base_accs))


def _event_model_objective(val_row: dict, val_zero_mae: float) -> float:
    """
    Validation-only objective.

    Prioritizes:
      1. active directional edge over same-days baselines
      2. small improvement over zero-return MAE
      3. not being absurdly tiny coverage
    """
    edge = _active_edge_vs_best_baseline(val_row)

    if not np.isfinite(edge):
        return -1e9

    mae = float(_row_get(val_row, "return_mae", np.nan))
    cov = float(_row_get(val_row, "active_coverage", 0.0))
    active_n = int(_row_get(val_row, "active_n", 0))

    if not np.isfinite(mae) or active_n <= 0:
        return -1e9

    mae_gain = (val_zero_mae - mae) / (abs(val_zero_mae) + 1e-12)

    # Mild penalty for ultra-small selected subsets.
    coverage_penalty = 0.02 * np.log(max(cov, 1e-6))

    return float(edge + 0.15 * mae_gain + coverage_penalty)


def _fit_event_model_candidate(
    train,
    val,
    test,
    feature_cols,
    selected_rules,
    top_k,
    horizon,
    config,
    model_name,
    model,
):
    chosen_rules = selected_rules[:top_k]

    train2 = add_selected_event_flags(train, chosen_rules)
    val2 = add_selected_event_flags(val, chosen_rules)
    test2 = add_selected_event_flags(test, chosen_rules)

    event_cols = [c for c in train2.columns if c.startswith("selected_event_")]

    if not event_cols:
        return None

    active_train = train2[event_cols].sum(axis=1).to_numpy() > 0
    active_val = val2[event_cols].sum(axis=1).to_numpy() > 0
    active_test_rule_union = test2[event_cols].sum(axis=1).to_numpy() > 0

    if active_train.sum() < config.model_min_samples:
        return None

    if active_val.sum() < config.min_val_event_n:
        return None

    target = f"target_direction_{horizon}"

    candidate_features = list(dict.fromkeys(event_cols + feature_cols))

    selected_features = select_model_features(
        train2.loc[active_train],
        candidate_features,
        target,
        config.model_max_features,
        config.seed + horizon + top_k,
    )

    if not selected_features:
        return None

    Xtr = train2.loc[active_train, selected_features]
    ytr = train2.loc[active_train, target].astype(int)

    model.fit(Xtr, ytr)

    p_val = np.full(len(val2), 0.5)
    p_test = np.full(len(test2), 0.5)

    if active_val.sum():
        p_val[active_val] = model.predict_proba(val2.loc[active_val, selected_features])[:, 1]

    if active_test_rule_union.sum():
        p_test[active_test_rule_union] = model.predict_proba(
            test2.loc[active_test_rule_union, selected_features]
        )[:, 1]

    # Inactive rows get neutral score so they cannot become active unless threshold is exactly 0.
    score_val = np.where(active_val, np.abs(p_val - 0.5), 0.0)
    score_test = np.where(active_test_rule_union, np.abs(p_test - 0.5), 0.0)

    thr_info = choose_confidence_threshold(
        val2,
        p_val,
        horizon,
        config.min_active_coverage,
        config.max_active_coverage,
    )

    scale = float(train[f"target_return_{horizon}"].abs().median())

    r_val = (p_val - 0.5) * 2 * scale
    r_test = (p_test - 0.5) * 2 * scale

    val_row = evaluate_probability_model(
        val2,
        p_val,
        horizon,
        "validation_event_model",
        f"{model_name}_top{top_k}",
        active_score=score_val,
        active_threshold=thr_info["threshold"],
        return_pred=r_val,
    )

    test_row = evaluate_probability_model(
        test2,
        p_test,
        horizon,
        "robust_event_model",
        f"{model_name}_top{top_k}",
        active_score=score_test,
        active_threshold=thr_info["threshold"],
        return_pred=r_test,
    )

    val_zero_mae = float(val2[f"target_return_{horizon}"].abs().mean())
    objective = _event_model_objective(val_row, val_zero_mae)

    active_test_final = score_test >= float(thr_info["threshold"])

    return {
        "objective": objective,
        "model_name": model_name,
        "top_k": top_k,
        "selected_features": selected_features,
        "val_row": val_row,
        "test_row": test_row,
        "p_test": p_test,
        "r_test": r_test,
        "score_test": score_test,
        "threshold": thr_info["threshold"],
        "active_test": active_test_final,
        "active_train_n": int(active_train.sum()),
        "active_val_n": int(active_val.sum()),
        "active_test_n_rule_union": int(active_test_rule_union.sum()),
    }


def train_event_models(train, val, test, feature_cols, selected_rules, selected_scores, horizon, config):
    rows = []
    pred_payload = {}

    if not selected_rules:
        return rows, pred_payload

    top_k_grid = getattr(config, "event_top_k_grid", (1, 2, 3, 5, 8, 12, 16))
    top_k_grid = sorted(
        set(
            int(k)
            for k in top_k_grid
            if int(k) >= 1 and int(k) <= len(selected_rules)
        )
    )

    models = {
        "hist_gb": _hist_model,
        "extra_trees": _extra_model,
        "logistic": _logistic_model,
    }

    best = None

    print(f"    validation-selecting event model over top_k grid: {top_k_grid}")

    for top_k in top_k_grid:
        for model_name, factory in models.items():
            model = factory(config.seed + horizon + top_k)

            try:
                candidate = _fit_event_model_candidate(
                    train=train,
                    val=val,
                    test=test,
                    feature_cols=feature_cols,
                    selected_rules=selected_rules,
                    top_k=top_k,
                    horizon=horizon,
                    config=config,
                    model_name=model_name,
                    model=model,
                )

                if candidate is None:
                    continue

                val_row = candidate["val_row"]

                print(
                    f"      {model_name:<11s} top_k={top_k:<2d} "
                    f"val_obj={candidate['objective']:.4f} "
                    f"val_active_acc={_row_get(val_row, 'active_directional_accuracy', np.nan):.3f} "
                    f"val_active_n={_row_get(val_row, 'active_n', 0)}"
                )

                if best is None or candidate["objective"] > best["objective"]:
                    best = candidate

            except Exception as e:
                print(f"      WARNING: candidate {model_name} top_k={top_k} failed: {e}")

    if best is None:
        print("    WARNING: no validation-selected event model survived.")
        return rows, pred_payload

    test_row = best["test_row"]

    test_row["model"] = f"{best['model_name']}_top{best['top_k']}_val_selected"
    test_row["selected_by"] = (
        f"validation_selected;"
        f"model={best['model_name']};"
        f"top_k={best['top_k']};"
        f"val_objective={best['objective']:.6f};"
        f"val_active_acc={_row_get(best['val_row'], 'active_directional_accuracy', np.nan):.6f};"
        f"val_active_n={_row_get(best['val_row'], 'active_n', 0)};"
        f"rule_union_train_n={best['active_train_n']};"
        f"rule_union_val_n={best['active_val_n']};"
        f"rule_union_test_n={best['active_test_n_rule_union']};"
        f"features={len(best['selected_features'])};"
        f"thr={best['threshold']:.6f}"
    )

    rows.append(test_row)

    key = f"robust_event_model_{best['model_name']}_top{best['top_k']}_h{horizon}"

    pred_payload[key] = {
        "prob": best["p_test"],
        "return_pred": best["r_test"],
        "score": best["score_test"],
        "threshold": best["threshold"],
        "active": best["active_test"],
    }

    print(
        f"    SELECTED: {best['model_name']} top_k={best['top_k']} "
        f"val_obj={best['objective']:.4f} "
        f"test_mae={test_row['return_mae']:.6f} "
        f"test_active_acc={_row_get(test_row, 'active_directional_accuracy', np.nan):.3f} "
        f"test_active_n={_row_get(test_row, 'active_n', 0)}"
    )

    return rows, pred_payload