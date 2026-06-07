from src.hierarchical_market_structure import LabConfig, run_experiment

config = LabConfig(mode="medium", ticker="BTC-USD", start="2016-01-01")
run_experiment(config)
