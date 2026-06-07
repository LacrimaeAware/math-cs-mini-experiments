from src.research_heavy_stack import ResearchConfig, run_research_heavy

# Middle version of the real stacked experiment.

config = ResearchConfig(
    mode="research_medium",
    ticker="BTC-USD",
    start="2016-01-01",
    horizons=(1, 2, 3),
    n_folds=4,
    max_base_features=220,
    include_specialists=True,
    include_big_tree=False,
    seed=42,
)

run_research_heavy(config)
