# Architecture Notes

This version is built around the idea that market structure should feed forward through a hierarchy rather than sitting in separate parallel experiments.

## Main pipeline

```text
raw OHLCV/context data
-> multi-scale feature extraction
-> regime/hypercategory construction
-> formula/interaction layer
-> specialist experts
-> validation-weighted final gate
```

## What is meant by hypercategories?

Examples:

- short-term dip inside long-term uptrend
- short-term dip inside long-term downtrend
- breakout inside long-term compression
- panic move inside long-term downtrend
- cyclic/choppy structure
- trend agreement or disagreement across scales

These categories are not just report labels. They become numeric features that feed formula generation, specialist models, and the final gate.

## What is meant by formula/hyperfunction layer?

The formula layer generates category-conditioned transformations such as:

- `z(feature) * regime_flag`
- short-scale feature minus long-scale feature
- short-scale feature times long-scale feature
- decayed weighted sums across time scales
- nonlinear transforms like `tanh`, absolute value, and squared terms

The formula layer is deliberately not a standalone model. Its selected formula features are appended to the feature table and used by global models, specialists, and the final gate.

## Specialist experts

Specialists are trained on subsets of the training data where a setup/regime is active. At prediction time, a flag-gated specialist ensemble decides which specialists are relevant to the current row.

## Final gate

The final gate receives predictions from baselines, global models, the direction model, specialists, and optional neural experts. It combines them using validation-weighted expert blending. This is intentionally conservative to reduce overfitting.

## Anti-overfitting measures

- chronological train/validation/test split
- train-only latent regime fitting
- validation-only formula/model/gate selection
- conservative tree depths and leaf sizes
- shrinkage of final gate toward zero return
- feature group ablation in medium/heavy modes
- active-set metrics reported with coverage

## Interpretation

A model that beats the zero-return/persistence baseline by a tiny amount should not be treated as a trading edge. The goal is to see whether the hierarchical structure produces any repeatable signal, especially for short-horizon direction and active/setup subsets.
