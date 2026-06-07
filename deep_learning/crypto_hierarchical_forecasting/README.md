# Crypto hierarchical forecasting

A BTC-USD forecasting experiment built around market structure and a
mixture-of-experts design rather than a single sequence model. This is the
market-structure line of the same project as the event robustness lab.

## Pipeline

```text
raw OHLCV / context data
-> multi-scale trend / dip / cycle / volatility features
-> regime and hypercategory flags
-> category-conditioned formulas and interactions
-> specialist expert models
-> final gated ensemble
```

Each market regime is routed to a specialist model, and the specialists are
combined with a gate, instead of scaling up one LSTM. The structured design is
compared against simple baselines and tabular ML. Architecture notes are in
[ARCHITECTURE_NOTES.md](ARCHITECTURE_NOTES.md).

## Run

```bash
pip install -r requirements.txt
python RUN_SYNTHETIC_SMOKE_TEST.py   # offline synthetic check
python RUN_QUICK.py                  # live BTC-USD, small config
python RUN_MEDIUM.py                 # main run
python RUN_HEAVY.py                  # larger config
```

The `RUN_RESEARCH_*` scripts run the heavier research stack
(`src/research_heavy_stack.py`). Runs write to a `hierarchical_market_outputs/`
directory; data is fetched from Yahoo Finance and not stored in the repository.

## Layout

```text
crypto_hierarchical_forecasting/
├── RUN_QUICK.py, RUN_MEDIUM.py, RUN_HEAVY.py, RUN_SYNTHETIC_SMOKE_TEST.py
├── RUN_RESEARCH_QUICK.py, RUN_RESEARCH_MEDIUM.py, RUN_RESEARCH_HEAVY.py
├── requirements.txt
├── ARCHITECTURE_NOTES.md
└── src/
    ├── hierarchical_market_structure.py   features, regimes, experts, gated ensemble
    └── research_heavy_stack.py            heavier research variant
```

## Status

A research and coursework experiment. As with the event line, the goal was to
test whether a structured design beats simple baselines on daily BTC-USD. The
results are exploratory, not a trading system.
