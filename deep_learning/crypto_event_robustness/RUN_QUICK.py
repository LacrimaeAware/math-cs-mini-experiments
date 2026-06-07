from src.config import preset
from src.experiment import run_experiment

if __name__ == "__main__":
    cfg = preset("quick")
    run_experiment(cfg)
