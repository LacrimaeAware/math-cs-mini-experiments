from src.research_heavy_stack import ResearchConfig, run_research_heavy

# This is intentionally much heavier than RUN_HEAVY.py.
# It performs expanding walk-forward stacking, trains base experts repeatedly,
# then trains a meta-model on out-of-fold expert predictions.

config = ResearchConfig(
    mode="research_heavy",
    ticker="BTC-USD",
    start="2016-01-01",
    horizons=(1, 2, 3, 5, 7),
    n_folds=6,
    max_base_features=320,
    include_specialists=True,
    include_big_tree=True,
    seed=42,
)

run_research_heavy(config)
