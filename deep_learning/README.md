# Deep learning

Machine-learning and deep-learning experiments. Original PDF write-ups and any
identifying information are not included; only the code and clean summaries are
kept here.

## Contents

- [`crypto_event_robustness/`](crypto_event_robustness/): a time-series
  event-mining and robustness-testing framework for BTC-USD. It mines candidate
  market events and keeps only the rules that survive same-day baselines, noise
  perturbation, and walk-forward checks. The result is a weak short-horizon
  directional signal, documented with its caveats rather than presented as a
  trading system.
- [`cnn_image_regression/`](cnn_image_regression/): two small CNN image-regression
  tasks (clock reading with a sin/cos circular target encoding, and triangle
  counting) on synthetically generated images.
- [`crypto_hierarchical_forecasting/`](crypto_hierarchical_forecasting/): a BTC-USD
  forecasting experiment using market-structure features and a mixture-of-experts
  gated ensemble (the market-structure line of the same project as the event
  robustness lab).

More experiments will be added.
