from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_outdir(config) -> Path:
    root = Path(config.output_root)
    root.mkdir(exist_ok=True)
    name = config.ticker.replace("-", "_").replace("^", "")
    import datetime as dt
    out = root / f"{name}_{config.mode}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_config(config, outdir: Path):
    d = dict(config.__dict__)
    for k, v in list(d.items()):
        if isinstance(v, tuple):
            d[k] = list(v)
    (outdir / "run_config.json").write_text(json.dumps(d, indent=2), encoding="utf-8")


def save_plots(outdir: Path, leaderboard: pd.DataFrame, candidates: pd.DataFrame, predictions: pd.DataFrame | None = None):
    if leaderboard is not None and not leaderboard.empty:
        top = leaderboard.sort_values("return_mae").head(20).copy()
        plt.figure(figsize=(10, 6))
        labels = [f"{r.track}\n{r.model}\nh{r.horizon}" for _, r in top.iterrows()]
        plt.barh(range(len(top)), top["return_mae"])
        plt.yticks(range(len(top)), labels, fontsize=7)
        plt.gca().invert_yaxis()
        plt.xlabel("Return MAE, lower is better")
        plt.title("Top models by return MAE")
        plt.tight_layout()
        plt.savefig(outdir / "01_return_mae_leaderboard.png", dpi=160)
        plt.close()

        topa = leaderboard.sort_values("active_directional_accuracy", ascending=False).head(20).copy()
        plt.figure(figsize=(10, 6))
        labels = [f"{r.track}\n{r.model}\nh{r.horizon}\nn={int(r.active_n)}" for _, r in topa.iterrows()]
        plt.barh(range(len(topa)), topa["active_directional_accuracy"])
        plt.yticks(range(len(topa)), labels, fontsize=7)
        plt.gca().invert_yaxis()
        plt.xlabel("Active directional accuracy")
        plt.title("Active-day directional accuracy")
        plt.tight_layout()
        plt.savefig(outdir / "02_active_directional_accuracy.png", dpi=160)
        plt.close()

    if candidates is not None and not candidates.empty:
        plt.figure(figsize=(8, 5))
        x = candidates["noise_survival"].fillna(0)
        y = candidates["val_edge"].fillna(0)
        s = np.clip(candidates["val_n"].fillna(1), 5, 300) / 2
        plt.scatter(x, y, s=s, alpha=0.6)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Noise survival fraction")
        plt.ylabel("Validation directional edge vs best baseline")
        plt.title("Candidate event robustness")
        plt.tight_layout()
        plt.savefig(outdir / "03_event_robustness_scatter.png", dpi=160)
        plt.close()

        topc = candidates.sort_values("robust_score", ascending=False).head(20).copy()
        plt.figure(figsize=(10, 6))
        labels = [str(x)[:60] for x in topc["rule_name"]]
        plt.barh(range(len(topc)), topc["robust_score"])
        plt.yticks(range(len(topc)), labels, fontsize=7)
        plt.gca().invert_yaxis()
        plt.xlabel("Robustness score")
        plt.title("Top discovered event rules")
        plt.tight_layout()
        plt.savefig(outdir / "04_top_event_rules.png", dpi=160)
        plt.close()

    if predictions is not None and not predictions.empty and "equity_best" in predictions.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(predictions["Date"], predictions["equity_best"], label="Toy strategy")
        if "equity_buy_hold" in predictions.columns:
            plt.plot(predictions["Date"], predictions["equity_buy_hold"], label="Buy and hold")
        plt.legend()
        plt.title("Toy equity curve, not used for model selection")
        plt.tight_layout()
        plt.savefig(outdir / "05_toy_equity_curve.png", dpi=160)
        plt.close()


def write_summary(outdir: Path, config, leaderboard: pd.DataFrame, candidates: pd.DataFrame, selected: pd.DataFrame):
    lines = []
    lines.append("# Event Robustness Lab Summary\n")
    lines.append(f"Mode: `{config.mode}`  ")
    lines.append(f"Ticker: `{config.ticker}`  ")
    lines.append(f"Horizons: `{list(config.horizons)}`  \n")
    lines.append("## Purpose\n")
    lines.append("This run searches for event/setup definitions and keeps patterns only if they survive validation, perturbation/noise tests, and walk-forward checks. The goal is not to predict every day, but to find event families that generalize better than simple same-days baselines.\n")
    if leaderboard is not None and not leaderboard.empty:
        lines.append("## Best rows by return MAE\n")
        lines.append("```text\n")
        cols = [c for c in ["track", "model", "horizon", "return_mae", "directional_accuracy", "active_n", "active_coverage", "active_directional_accuracy", "same_active_zero_acc", "same_active_prev_same_h_acc", "same_active_ret_mean_60_acc"] if c in leaderboard.columns]
        lines.append(leaderboard.sort_values("return_mae").head(15)[cols].to_string(index=False))
        lines.append("\n```\n")
        lines.append("## Best rows by active directional accuracy\n")
        lines.append("```text\n")
        lines.append(leaderboard.sort_values("active_directional_accuracy", ascending=False).head(15)[cols].to_string(index=False))
        lines.append("\n```\n")
    if selected is not None and not selected.empty:
        lines.append("## Selected robust event rules\n")
        lines.append("```text\n")
        cols = [c for c in ["rule_name", "horizon", "source", "train_n", "val_n", "val_edge", "noise_survival", "wf_positive_frac", "robust_score"] if c in selected.columns]
        lines.append(selected[cols].head(30).to_string(index=False))
        lines.append("\n```\n")
    lines.append("## Reading the results\n")
    lines.append("- `return_mae` compares return magnitude predictions; zero-return is usually very hard to beat.\n")
    lines.append("- `active_directional_accuracy` only measures rows where the model made an active/high-confidence call.\n")
    lines.append("- Same-active baseline columns compare zero, momentum, and previous-move baselines on exactly the same active days.\n")
    lines.append("- A useful event should beat these same-days baselines and have enough active samples.\n")
    (outdir / "REPORT_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
