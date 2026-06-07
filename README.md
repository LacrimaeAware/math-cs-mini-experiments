# MathExperimentation

Experiments and notes across mathematics and adjacent code, organized by domain
so more areas can be added over time. The current content is mostly number theory
(primes, primorials, twin primes, modular sieves), with machine-learning and
deep-learning experiments and a few plotting scripts. Each script is standalone and prints or
plots its output. Shared primality, sieve, and primorial helpers are in
`prime_lib.py`. Generated files are written to `outputs/`.

## Layout

```text
MathExperimentation/
├── prime_lib.py                shared helpers: primality tests, sieves, primorials, output paths
├── main.py                     entry stub
├── number_theory/              number-theory scripts and notes
│   ├── README.md               area index and notes-to-scripts map
│   ├── notes/twin_primes/      twin-prime notes, one topic per file, with an index
│   ├── primorials_and_twin_primes/
│   ├── primorial_wheels/
│   ├── prime_set_constructions/
│   └── prime_patterns/
├── ml_and_visuals/             logistic-regression animation, MNIST setup, 3D region plot
├── deep_learning/              ML / deep-learning experiments (crypto event robustness lab, etc.)
├── docs/
│   ├── research_overview.md    per-script description of the number-theory ideas
│   └── conventions.md          writing conventions for this repository
├── outputs/                    generated images, data, and caches
├── pyproject.toml, uv.lock     dependencies (uv)
└── private/                    git-ignored notes (not published)
```

## Number-theory scripts

Paths are under `number_theory/`.

| Script | Computes |
|---|---|
| `primorials_and_twin_primes/plus_or_minus_primorial_squares.py` | Factors `p_n# +/- offset` and tests whether a factor belongs to a twin-prime pair (cached). |
| `primorials_and_twin_primes/primorial_radius_scan_twin_prime_factor.py` | Scans offsets `d = 1..R` around a primorial and ranks offsets by twin-prime-factor rate. |
| `primorials_and_twin_primes/primorial_to_prime_distance_run_ratios.py` | First composite distance from a primorial, by bidirectional nearest-prime search. |
| `primorials_and_twin_primes/primorial_center_radii_twinp.py` | Exact rational expressions around `m = nearest_int(P/A^2)`; factors `m +/- B`. |
| `primorial_wheels/primorial_gap_freq.py` | Border-merge gap statistics when a wheel is sieved by its next prime. |
| `primorial_wheels/primorial_gap_density_count.py` | Worst-case deviation of a model period for the residue classes in a primorial modulus. |
| `primorial_wheels/chunks_of_x_symm.py` | Coprime counts per fixed-size block of a primorial. |
| `primorial_wheels/chunks_of_6_symm.py` | Coprime residues mod 6 (the `6k +/- 1` structure) across `[1, N]`. |
| `primorial_wheels/orbit_primorials.py`, `extended_orbits.py` | Reconstruct the coprimes of a primorial via `C +/- P^x` orbits; compare to `phi(N)`. |
| `primorial_wheels/nonlinear_arithmetic_sets.py` | Signed-sum sets covering the coprimes mod 210. |
| `prime_set_constructions/combinatorics_lemma.py` (+ `_exponents`, `_exhaustive`) | Omission-term / CRT-basis construction; counts primes among signed sums. |
| `prime_set_constructions/clustering_prime_products.py` | Branch-and-bound partition of the first `k` primes by product balance. |
| `prime_patterns/head_tail.py` | Primality of `A*base^k + X` over `(A, k)` grids vs. the expected count `sum 1/log N`; writes heatmaps. |
| `prime_patterns/prime_deadends.py` | Digit-prepending search for left-truncatable prime chains. |
| `prime_patterns/prime_squares_distance_log_terms.py` | Fraction of consecutive `q^2 - p^2` divisible by 5 (reads `primes.txt`). |

The idea behind each script is in [docs/research_overview.md](docs/research_overview.md).
The twin-prime notes are in [number_theory/notes/twin_primes/](number_theory/notes/twin_primes/).

## Machine-learning and plotting scripts

| Script | Output |
|---|---|
| `ml_and_visuals/logistic_regression_animation.py` | Animation of logistic-regression gradient descent (reads ffmpeg from `FFMPEG_PATH` or `PATH`). |
| `ml_and_visuals/mnist_keras_intro.py` | MNIST with a dense Keras model; sets up the model but does not call `fit`. |
| `ml_and_visuals/region_plot_3d.py` | Interactive 3D plot of a region (requires `plotly`). |

## Run

Uses [uv](https://docs.astral.sh/uv/) (Python 3.13).

```bash
uv sync
uv run python number_theory/primorial_wheels/chunks_of_x_symm.py --pmax 19 --block 510510
uv run python number_theory/primorial_wheels/primorial_gap_freq.py --primes 2 3 5 7 --next 11
```

Scripts locate the repository root to import `prime_lib` and write to `outputs/`,
so they run from any directory. Most scripts have a configuration block (constants
or argparse) at the top.

## Notes

- `prime_patterns/prime_squares_distance_log_terms.py` reads a `primes.txt` (one
  prime per line) that is not included.
- `region_plot_3d.py` requires `plotly`, which is not listed in `pyproject.toml`.
- `mnist_keras_intro.py` stops before training.
- `nonlinear_arithmetic_sets.py` starts its search on import (no `__main__` guard).
- Large-integer primality uses probabilistic Miller-Rabin (`prime_lib.is_probable_prime`).
- Writing conventions for this repository are in [docs/conventions.md](docs/conventions.md).
