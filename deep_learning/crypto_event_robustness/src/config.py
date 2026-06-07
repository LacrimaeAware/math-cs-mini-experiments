from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    mode: str = "quick"
    ticker: str = "BTC-USD"
    start: str = "2016-01-01"
    end: str | None = None
    output_root: str = "event_robustness_outputs"
    seed: int = 42
    use_synthetic: bool = False
    local_csv: str | None = None

    context_tickers: tuple[str, ...] = (
        "ETH-USD",
        "SPY",
        "QQQ",
        "GLD",
        "TLT",
        "DX-Y.NYB",
        "^VIX",
    )

    train_frac: float = 0.70
    val_frac: float = 0.15
    horizons: tuple[int, ...] = (1, 3)
    cost_bps: float = 10.0

    scales: tuple[int, ...] = (2, 3, 5, 7, 14, 21, 30, 60, 90, 120)
    fft_windows: tuple[int, ...] = (30, 60, 120)
    learned_event_windows: tuple[int, ...] = (30, 60)
    learned_event_clusters: int = 8
    learned_event_pca_components: int = 8

    # Event mining controls
    max_candidate_rules: int = 250
    quantiles: tuple[float, ...] = (0.10, 0.20, 0.30, 0.70, 0.80, 0.90)
    max_pair_rules: int = 120
    min_train_event_n: int = 80
    min_val_event_n: int = 20
    min_test_event_n: int = 20

    # Active prediction controls.
    # These are intentionally conservative now to avoid selecting tiny validation pockets.
    min_active_coverage: float = 0.06
    max_active_coverage: float = 0.35

    perturb_trials: int = 35
    perturb_noise_scale: float = 0.10
    perturb_threshold_scale: float = 0.10
    walk_forward_folds: int = 4
    walk_forward_gap: int = 3
    selected_event_count: int = 15
    event_top_k_grid: tuple[int, ...] = (1, 2, 3, 5, 8)
    min_validation_edge: float = -0.01
    min_positive_fold_frac: float = 0.25

    model_max_features: int = 100
    model_min_samples: int = 80
    run_event_specific_models: bool = True
    run_global_event_model: bool = True

    include_learned_events: bool = True
    include_manual_seed_events: bool = True
    include_threshold_rules: bool = True
    include_pair_rules: bool = True

    # Toy trading simulation, not used for model selection.
    initial_cash: float = 1000.0
    max_position: float = 1.0

    def output_dir(self) -> Path:
        return Path(self.output_root)


def preset(mode: str) -> ExperimentConfig:
    mode = mode.lower().strip()
    cfg = ExperimentConfig(mode=mode)

    if mode == "quick":
        cfg.horizons = (1, 3)
        cfg.scales = (2, 3, 7, 14, 30, 60)
        cfg.fft_windows = (30, 60)
        cfg.learned_event_windows = (30,)
        cfg.learned_event_clusters = 6
        cfg.max_candidate_rules = 90
        cfg.max_pair_rules = 25
        cfg.perturb_trials = 10
        cfg.walk_forward_folds = 3
        cfg.selected_event_count = 8
        cfg.event_top_k_grid = (1, 2, 3, 5, 8)
        cfg.model_max_features = 80
        cfg.min_active_coverage = 0.06
        cfg.max_active_coverage = 0.35

    elif mode == "medium":
        cfg.horizons = (1, 3)
        cfg.scales = (2, 3, 5, 7, 14, 21, 30, 60, 90, 120)
        cfg.fft_windows = (30, 60, 120)
        cfg.learned_event_windows = (30, 60)
        cfg.learned_event_clusters = 8
        cfg.max_candidate_rules = 220
        cfg.max_pair_rules = 80
        cfg.perturb_trials = 30
        cfg.walk_forward_folds = 5

        # Conservative event selection:
        # do not allow the model to combine too many events or chase tiny validation pockets.
        cfg.selected_event_count = 8
        cfg.event_top_k_grid = (1, 2, 3, 5, 8)
        cfg.model_max_features = 80
        cfg.min_active_coverage = 0.10
        cfg.max_active_coverage = 0.35

    elif mode == "heavy":
        cfg.horizons = (1, 2, 3, 5)
        cfg.scales = (1, 2, 3, 5, 7, 14, 21, 30, 60, 90, 120, 180, 240)
        cfg.fft_windows = (30, 60, 120, 180)
        cfg.learned_event_windows = (30, 60, 120)
        cfg.learned_event_clusters = 10
        cfg.max_candidate_rules = 450
        cfg.max_pair_rules = 180
        cfg.perturb_trials = 60
        cfg.walk_forward_folds = 6

        # Conservative heavy:
        # more candidates are allowed, but final event/model selection stays constrained.
        cfg.selected_event_count = 12
        cfg.event_top_k_grid = (1, 2, 3, 5, 8, 12)
        cfg.model_max_features = 100
        cfg.min_train_event_n = 100
        cfg.min_active_coverage = 0.10
        cfg.max_active_coverage = 0.35

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return cfg