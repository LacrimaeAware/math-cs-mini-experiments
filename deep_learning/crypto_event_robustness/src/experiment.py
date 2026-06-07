from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import load_market_data
from .features.assembly import build_feature_frame, split_chronological, infer_feature_columns, feature_groups
from .events.rules import generate_candidate_rules
from .events.scoring import score_candidate_rules
from .models.event_models import prior_event_ensemble, train_event_models
from .evaluation.metrics import evaluate_probability_model, baseline_returns
from .evaluation.backtest import toy_long_cash_equity
from .reporting import ensure_outdir, write_config, save_plots, write_summary


def _print_header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _baseline_rows(df: pd.DataFrame, horizon: int):
    rows = []

    for name, ret_pred in baseline_returns(df, horizon).items():
        prob = np.where(ret_pred > 0, 0.55, np.where(ret_pred < 0, 0.45, 0.5))

        row = evaluate_probability_model(
            df,
            prob,
            horizon,
            "baseline",
            name,
            active_score=np.ones(len(df)),
            active_threshold=0.0,
            return_pred=ret_pred,
        )

        row["selected_by"] = "baseline"
        rows.append(row)

    return rows


def _choose_selected_rules(candidate_scores: pd.DataFrame, rule_map: dict[str, object], config):
    if candidate_scores is None or candidate_scores.empty:
        return [], pd.DataFrame()

    filt = candidate_scores.copy()

    filt = filt[
        (filt["val_n"] >= config.min_val_event_n)
        & (filt["train_n"] >= config.min_train_event_n)
    ]

    filt = filt[filt["val_edge"] >= config.min_validation_edge]

    if "wf_positive_frac" in filt:
        filt = filt[filt["wf_positive_frac"].fillna(0) >= config.min_positive_fold_frac]

    filt = (
        filt.sort_values("robust_score", ascending=False)
        .head(config.selected_event_count)
        .copy()
    )

    selected = [rule_map[n] for n in filt["rule_name"] if n in rule_map]

    return selected, filt


def _candidate_union_table(all_candidate_scores: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [x for x in all_candidate_scores if x is not None and not x.empty]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=0, ignore_index=True)


def _print_selected_event_table(selected_table: pd.DataFrame, n: int = 12):
    if selected_table is None or selected_table.empty:
        print("  No selected event table to print.")
        return

    wanted = [
        "rule_name",
        "source",
        "train_n",
        "val_n",
        "val_coverage",
        "val_edge",
        "noise_survival",
        "wf_positive_frac",
        "robust_score",
    ]

    cols = [c for c in wanted if c in selected_table.columns]

    print(selected_table[cols].head(n).to_string(index=False))


def _print_leaderboard_table(leaderboard: pd.DataFrame, sort_col: str, title: str, n: int = 20):
    if leaderboard.empty:
        return

    _print_header(title)

    cols = [
        "track",
        "model",
        "horizon",
        "return_mae",
        "directional_accuracy",
        "active_n",
        "active_coverage",
        "active_directional_accuracy",
        "same_active_zero_acc",
        "same_active_prev_same_h_acc",
        "same_active_ret_mean_60_acc",
        "selected_by",
    ]

    cols = [c for c in cols if c in leaderboard.columns]

    if sort_col in leaderboard.columns:
        shown = leaderboard.sort_values(sort_col, ascending=False if sort_col == "active_directional_accuracy" else True)
    else:
        shown = leaderboard

    print(shown[cols].head(n).to_string(index=False))


def run_experiment(config):
    _print_header("Crypto Event Robustness Lab")
    print(f"Mode: {config.mode}")
    print(f"Ticker: {config.ticker}")
    print(f"Horizons: {list(config.horizons)}")
    print("Goal: discover event/setup rules and keep only patterns that survive perturbation and walk-forward checks.")

    outdir = ensure_outdir(config)
    write_config(config, outdir)
    print(f"Output directory: {outdir}")

    _print_header("Loading data")
    raw = load_market_data(config)
    print(f"Rows loaded: {len(raw)} | Date range: {raw.index.min().date()} to {raw.index.max().date()}")

    _print_header("Feature engineering")
    feat = build_feature_frame(raw, config)
    print(f"Columns after feature engineering before split: {len(feat.columns)}")

    train, val, test = split_chronological(feat, config)

    print("Chronological split")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s}: rows={len(part):5d} | {part.index.min().date()} to {part.index.max().date()}")

    feature_cols = infer_feature_columns(pd.concat([train, val, test], axis=0))
    groups = feature_groups(feature_cols)

    print(f"Usable feature count: {len(feature_cols)}")
    for k, v in groups.items():
        print(f"  {k:18s}: {len(v)}")

    _print_header("Generating candidate events")
    rules = generate_candidate_rules(train, feature_cols, config)
    rule_map = {r.name: r for r in rules}
    print(f"Candidate event rules generated: {len(rules)}")

    all_rows = []
    all_candidate_scores = []
    all_selected_tables = []

    prediction_table = pd.DataFrame({
        "Date": test.index,
        "Close": test["Close"].to_numpy(),
    })

    for horizon in config.horizons:
        _print_header(f"Horizon {horizon} days")

        # Baselines on full test.
        base_rows = _baseline_rows(test, horizon)

        for r in base_rows:
            print(
                f"baseline {r['model']:<24s} | "
                f"MAE={r['return_mae']:.6f} | "
                f"dir={r['directional_accuracy']:.3f}"
            )

        all_rows.extend(base_rows)

        print("Scoring candidate events with validation, noise perturbation, and walk-forward checks...")

        cand = score_candidate_rules(train, val, rules, horizon, config)

        if cand.empty:
            print("  No candidate events survived minimum sample/coverage filters.")
            continue

        cand.to_csv(outdir / f"candidate_scores_h{horizon}.csv", index=False)
        all_candidate_scores.append(cand)

        selected_rules, selected_table = _choose_selected_rules(cand, rule_map, config)

        if selected_table.empty:
            print("  No robust events selected after robustness filters; using top robustness events as fallback.")
            selected_table = (
                cand.sort_values("robust_score", ascending=False)
                .head(min(config.selected_event_count, len(cand)))
                .copy()
            )
            selected_rules = [rule_map[n] for n in selected_table["rule_name"] if n in rule_map]

        selected_table.to_csv(outdir / f"selected_events_h{horizon}.csv", index=False)
        all_selected_tables.append(selected_table)

        print(f"Selected robust event rules: {len(selected_rules)}")
        _print_selected_event_table(selected_table, n=12)

        prior_rows, prior_preds = prior_event_ensemble(
            train,
            val,
            test,
            selected_rules,
            selected_table["robust_score"].to_list(),
            horizon,
            config.cost_bps,
        )

        all_rows.extend(prior_rows)

        model_rows, model_preds = train_event_models(
            train,
            val,
            test,
            feature_cols,
            selected_rules,
            selected_table["robust_score"].to_list(),
            horizon,
            config,
        )

        all_rows.extend(model_rows)

        # Save prediction columns for toy equity for the best later.
        for name, payload in {**prior_preds, **model_preds}.items():
            prediction_table[f"prob_{name}"] = payload["prob"]
            prediction_table[f"active_{name}"] = payload["active"]

            if "return_pred" in payload:
                prediction_table[f"retpred_{name}"] = payload["return_pred"]

            if horizon == 1:
                prediction_table[f"equity_{name}"] = toy_long_cash_equity(
                    test,
                    payload["prob"],
                    payload["active"],
                    horizon,
                    config.cost_bps,
                    config.initial_cash,
                )

    leaderboard = pd.DataFrame(all_rows)

    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values("return_mae").reset_index(drop=True)
        leaderboard.to_csv(outdir / "leaderboard.csv", index=False)

        _print_leaderboard_table(
            leaderboard,
            sort_col="return_mae",
            title="Best rows by return MAE",
            n=20,
        )

        _print_leaderboard_table(
            leaderboard,
            sort_col="active_directional_accuracy",
            title="Best rows by active directional accuracy",
            n=20,
        )
    else:
        print("No leaderboard rows produced.")

    candidate_all = _candidate_union_table(all_candidate_scores)

    if not candidate_all.empty:
        candidate_all.to_csv(outdir / "event_candidates_scored.csv", index=False)

    selected_all = _candidate_union_table(all_selected_tables)

    if not selected_all.empty:
        selected_all.to_csv(outdir / "selected_events.csv", index=False)

    # Buy/hold equity for h=1 context and best toy curve.
    if not prediction_table.empty:
        if "target_return_1" in test.columns:
            prediction_table["equity_buy_hold"] = config.initial_cash * np.exp(
                np.cumsum(test["target_return_1"].to_numpy(dtype=float))
            )

        eq_cols = [
            c for c in prediction_table.columns
            if c.startswith("equity_") and c not in ["equity_buy_hold"]
        ]

        if eq_cols:
            best_eq_col = max(eq_cols, key=lambda c: prediction_table[c].iloc[-1])
            prediction_table["equity_best"] = prediction_table[best_eq_col]
            prediction_table["best_equity_model"] = best_eq_col

        prediction_table.to_csv(outdir / "predictions_and_equity.csv", index=False)

    save_plots(outdir, leaderboard, candidate_all, prediction_table)
    write_summary(outdir, config, leaderboard, candidate_all, selected_all)

    _print_header("Saved outputs")
    print(outdir)
    print("Open these first:")

    for fn in [
        "REPORT_SUMMARY.md",
        "leaderboard.csv",
        "event_candidates_scored.csv",
        "selected_events.csv",
        "predictions_and_equity.csv",
        "01_return_mae_leaderboard.png",
        "03_event_robustness_scatter.png",
    ]:
        p = outdir / fn
        if p.exists():
            print(f"  {p}")

    return outdir