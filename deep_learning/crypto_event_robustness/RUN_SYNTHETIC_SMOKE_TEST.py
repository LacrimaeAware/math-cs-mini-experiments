from src.config import preset
from src.experiment import run_experiment

if __name__ == "__main__":
    cfg = preset("quick")
    cfg.mode = "synthetic_smoke_test"
    cfg.use_synthetic = True
    cfg.output_root = "synthetic_event_robustness_outputs"
    cfg.perturb_trials = 8
    cfg.walk_forward_folds = 2
    cfg.max_candidate_rules = 80
    cfg.max_pair_rules = 20
    cfg.selected_event_count = 8
    run_experiment(cfg)
