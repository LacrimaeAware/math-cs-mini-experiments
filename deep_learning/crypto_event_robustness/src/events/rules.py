from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class Term:
    feature: str
    op: str
    threshold: float = 0.5
    std: float = 1.0

    def eval(self, df: pd.DataFrame, rng=None, noise_scale: float = 0.0, threshold_scale: float = 0.0) -> np.ndarray:
        x = df[self.feature].to_numpy(dtype=float)
        thr = float(self.threshold)
        std = float(self.std) if np.isfinite(self.std) and self.std > EPS else 1.0
        if rng is not None and noise_scale > 0 and self.feature in df.columns:
            x = x + rng.normal(0.0, noise_scale * std, size=len(x))
        if rng is not None and threshold_scale > 0 and self.op in ("<=", ">"):
            thr = thr + rng.normal(0.0, threshold_scale * std)
        if self.op == "<=":
            return np.isfinite(x) & (x <= thr)
        if self.op == ">":
            return np.isfinite(x) & (x > thr)
        if self.op == "flag":
            return np.isfinite(x) & (x > 0.5)
        raise ValueError(f"Unknown op: {self.op}")


@dataclass(frozen=True)
class EventRule:
    name: str
    terms: tuple[Term, ...]
    source: str = "mined"

    def eval(self, df: pd.DataFrame, rng=None, noise_scale: float = 0.0, threshold_scale: float = 0.0) -> np.ndarray:
        if not self.terms:
            return np.zeros(len(df), dtype=bool)
        mask = np.ones(len(df), dtype=bool)
        for t in self.terms:
            if t.feature not in df.columns:
                return np.zeros(len(df), dtype=bool)
            mask &= t.eval(df, rng=rng, noise_scale=noise_scale, threshold_scale=threshold_scale)
        return mask


def _safe_std(s: pd.Series) -> float:
    v = float(s.std(skipna=True))
    return v if np.isfinite(v) and v > EPS else 1.0


def _slug(x: str) -> str:
    return x.replace("<=", "le").replace(">", "gt").replace("-", "m").replace(".", "p")


def generate_candidate_rules(train: pd.DataFrame, feature_cols: list[str], config) -> list[EventRule]:
    rng = np.random.default_rng(config.seed)
    rules: list[EventRule] = []

    # Manual and learned flags become candidate events.
    flag_cols = []
    if getattr(config, "include_manual_seed_events", True):
        flag_cols += [c for c in feature_cols if c.startswith("seed_")]
    if getattr(config, "include_learned_events", True):
        flag_cols += [c for c in feature_cols if c.startswith("learned_") and ("evt_" in c or c.endswith("_changed"))]
    for c in flag_cols[:120]:
        if train[c].fillna(0).mean() > 0.005:
            rules.append(EventRule(name=f"flag__{c}", terms=(Term(c, "flag", 0.5, 1.0),), source="flag"))

    if getattr(config, "include_threshold_rules", True):
        # Feature types most likely to define event boundaries. Exclude raw price level.
        prefixes = ("ret_sum_", "trend_slope_", "drawdown_from_high_", "bounce_from_low_", "price_position_", "ret_std_", "vol_compression_", "fft_", "band_", "dow_", "month_")
        threshold_features = [c for c in feature_cols if c.startswith(prefixes) and train[c].notna().sum() > 200]
        # Favor interpretable/compact features.
        threshold_features = sorted(threshold_features, key=lambda c: (0 if any(k in c for k in ["ret_sum", "drawdown", "price_position", "trend_slope"]) else 1, len(c)))
        threshold_features = threshold_features[:80]
        for c in threshold_features:
            s = train[c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 200 or s.nunique() < 10:
                continue
            std = _safe_std(s)
            for q in config.quantiles:
                thr = float(s.quantile(q))
                op = "<=" if q < 0.5 else ">"
                name = f"thr__{_slug(c)}__{op}_{q:.2f}"
                rules.append(EventRule(name=name, terms=(Term(c, op, thr, std),), source="threshold"))

    if getattr(config, "include_pair_rules", True):
        # Build targeted pair rules: dip/trend, dip/volatility, frequency/trend, day/event.
        singles = [r for r in rules if len(r.terms) == 1 and r.source in ("threshold", "flag")]
        def cat(rule):
            f = rule.terms[0].feature
            if "drawdown" in f or "price_position" in f or "bounce" in f or "ret_sum_" in f:
                return "move"
            if "trend" in f or "ma_gap" in f:
                return "trend"
            if "std" in f or "vol" in f or "compression" in f:
                return "vol"
            if f.startswith("fft_") or f.startswith("band_"):
                return "freq"
            if f.startswith("seed_") or f.startswith("learned_"):
                return "event"
            if f.startswith("dow_") or f.startswith("month_"):
                return "cal"
            return "other"
        buckets = {}
        for r in singles:
            buckets.setdefault(cat(r), []).append(r)
        pair_sources = []
        for a, b in [("move", "trend"), ("move", "vol"), ("freq", "trend"), ("event", "trend"), ("event", "move"), ("cal", "move")]:
            for r1 in buckets.get(a, [])[:30]:
                for r2 in buckets.get(b, [])[:30]:
                    pair_sources.append((r1, r2))
        rng.shuffle(pair_sources)
        for r1, r2 in pair_sources[:config.max_pair_rules]:
            name = f"pair__{r1.name[:45]}__AND__{r2.name[:45]}"
            rules.append(EventRule(name=name, terms=(r1.terms[0], r2.terms[0]), source="pair"))

    # De-duplicate by name and cap.
    seen = set(); unique = []
    for r in rules:
        if r.name not in seen:
            seen.add(r.name); unique.append(r)
    return unique[:config.max_candidate_rules]
