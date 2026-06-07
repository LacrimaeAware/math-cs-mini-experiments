# Crypto event robustness lab

A time-series event-mining and robustness-testing framework for BTC-USD daily
data. It does not forecast every day. It mines candidate market events (setups)
and keeps only the rules that survive same-day baselines, noise perturbation, and
walk-forward checks. The purpose is to test whether apparent market signals
generalize, not to provide a trading system.

A detailed write-up (architecture, per-module detail, a consolidated results
table, lessons, a validation roadmap, and references) is in
[REPORT.md](REPORT.md).

## Pipeline

```text
market data (yfinance: BTC-USD plus context tickers)
-> normalized multi-scale features
-> manual seed events + learned cluster event ingredients
-> candidate event/rule mining
-> validation scoring vs same-day baselines
-> noise-perturbation robustness checks
-> walk-forward consistency checks
-> selected robust event rules
-> event-specific models and a toy long/cash simulation
```

## Baselines

Each model is graded against same-day baselines, so a signal has to beat them
rather than beat nothing: zero-return, previous-direction, and 60-day mean-return
momentum. The zero-return baseline is strong for daily crypto returns.

## Results

The event-selected models beat the zero-return MAE baseline several times, by
small margins, mostly at short horizons.

| Run | Horizon | Model MAE | Zero MAE | Relative MAE gain | Active dir. accuracy | Active n |
|---|---|---|---|---|---|---|
| quick | 3 | 0.029701 | 0.029863 | 0.54% | 61.1% | 36 |
| medium | 3 | 0.029677 | 0.029865 | 0.63% | (low active n) | 13 |
| heavy | 2 | 0.024286 | 0.024364 | 0.32% | 60.66% | 61 |

Caveats:

- The MAE gains are small (about 0.016% to 0.019% of return-error units at
  horizon 3), below a 10 bps transaction-cost setting.
- The heavy horizon-2 active accuracy (60.66%) barely exceeds the 60-day momentum
  baseline (59.02%) on the same active days.
- Several runs have small active sample counts (medium horizon 3 has active n = 13).
- Many models, horizons, and event rules were tested against the same test period,
  so the results carry a research-overfitting risk: repeated human selection can
  overfit even when the code does not leak.

The recurring event family was pullback / near-low / recent-weakness /
price-position-low setups, a plausible market condition rather than a random rule.
The evidence supports a weak short-horizon directional signal, not a reliable
trading edge. Confirming an edge would require positive return after costs on a
frozen, untouched test period, or on other assets and time periods.

## Run

```bash
pip install -r requirements.txt
python RUN_SYNTHETIC_SMOKE_TEST.py   # offline synthetic data, fast check
python RUN_QUICK.py                  # live BTC-USD, small config
python RUN_MEDIUM.py                 # main run
python RUN_HEAVY.py                  # larger config
```

Each run writes to `event_robustness_outputs/<ticker>_<mode>_<timestamp>/`:
`REPORT_SUMMARY.md`, `leaderboard.csv`, `event_candidates_scored.csv`,
`selected_events.csv`, `predictions_and_equity.csv`, and plots. Data is fetched
from Yahoo Finance; no market data is stored in the repository.

## Layout

```text
crypto_event_robustness/
├── RUN_QUICK.py, RUN_MEDIUM.py, RUN_HEAVY.py, RUN_SYNTHETIC_SMOKE_TEST.py
├── requirements.txt
└── src/
    ├── config.py            run presets (quick/medium/heavy) and all knobs
    ├── data_loader.py       yfinance loader with synthetic and local-CSV fallbacks
    ├── targets.py           forward log-return targets
    ├── features/            basic, multiscale, frequency, learned-cluster, event features
    ├── events/              candidate rule generation and robustness scoring
    ├── models/              event-specific and global event models
    ├── evaluation/          metrics, baselines, toy long/cash backtest
    ├── experiment.py        orchestration
    └── reporting.py         summary, plots, CSV outputs
```

## Limitations

- Daily BTC-USD from public Yahoo Finance data with technical and context
  features. MAE improvement is not profit: direction accuracy, move size, position
  sizing, fees, slippage, and drawdown are not modeled end to end.
- The signal, if real, is weak and short-horizon. Better structure or data
  (relative value, funding rates, liquidations, order-book imbalance, on-chain
  flows, or a multi-asset basket) is more likely to matter than further model
  tuning.
