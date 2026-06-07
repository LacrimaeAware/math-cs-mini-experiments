from src.research_heavy_stack import ResearchConfig, run_research_heavy

# Quick sanity version of the true stacked experiment.

config = ResearchConfig(
    mode="research_quick",
    ticker="BTC-USD",
    start="2016-01-01",
    horizons=(1, 3),
    n_folds=3,
    max_base_features=160,
    include_specialists=True,
    include_big_tree=False,
    seed=42,
)

run_research_heavy(config)
