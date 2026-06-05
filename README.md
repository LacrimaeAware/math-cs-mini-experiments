# MathExperimentation

A personal **experimental number-theory playground** — a collection of
self-contained Python scripts probing the structure of **primes and primorials**:
their distribution, gaps, coprime/"wheel" structure, twin-prime factors near
primorials, and prime-generating set constructions. Plus a couple of unrelated
machine-learning / visualization side-experiments.

Nothing here is a polished library; it's a lab notebook in code. Each script
runs standalone and prints (or plots) what it found. Shared primality/sieve
helpers live in [`prime_lib.py`](prime_lib.py); generated artifacts go to
[`outputs/`](outputs/).

**New here?** Start with **[`docs/research_overview.md`](docs/research_overview.md)** —
it ties each experiment to the underlying idea (modular sieves, primorial
structure, twin-prime constraints) and explains why the scripts exist.

---

## Layout

```
prime_lib.py                  # shared helpers: primality tests, sieves, primorials, output paths
main.py                       # tiny entry stub (just prints a pointer to this README)

primorials_and_twin_primes/   # behavior of primes in the neighborhood of a primorial
primorial_wheels/             # coprime / residue / gap structure of primorial moduli
prime_set_constructions/      # building primes from signed products of small-prime sets
prime_patterns/               # primality scans over families of integers
ml_and_visuals/               # the "weird stuff": ML + plotting side-quests

docs/                         # research_overview.md — how the experiments tie together
outputs/                      # all generated images, data, caches (gallery + records)
private/                      # (git-ignored) long-form research notes, not for publication
pyproject.toml / uv.lock      # dependencies (managed with uv)
```

---

## The experiments

### `primorials_and_twin_primes/` — primes near a primorial
- **`plus_or_minus_primorial_squares.py`** — for the n-th primorial `p_n#`, factor
  `p_n# ± offset` and check whether any factor is part of a *twin* prime pair.
  Cached + logged to `outputs/primorial_squares_cache/`.
- **`Primorial_radius_scan_twin_prime_factor.py`** — same idea, sweeping a whole
  radius of offsets `d = 1..R` and ranking which offsets most often yield a
  twin-prime factor.
- **`Primorial_To_Prime_Distance_Run_Ratios.py`** — "first composite distance"
  from a primorial via a bidirectional nearest-prime search.
- **`Primorial_Center_Radii_TwinP.py`** — exact rational expressions around the
  midpoint `m ≈ P/A²`, factoring `m ± B`.

### `primorial_wheels/` — coprime & gap structure
- **`PrimorialGapDensityCount.py`** — worst-case deviation of a model "period" for
  the residue classes of each prime inside a primorial modulus.
- **`PrimorialGapFreq.py`** — wheel "border-merge" gap statistics: which gaps merge
  with which when you sieve a wheel by its next prime. (CLI: `--primes 2 3 5 7 --next 11`)
- **`orbit_primorials.py`** / **`extended_orbits.py`** — reconstruct the coprimes of
  `N` via `C ± Pˣ` "orbits" over squarefree divisors `P`, compared against Euler's φ(N).
- **`Chunks_of_6_symm.py`** — visualize totative residues mod 6 (the `6k ± 1` structure).
- **`Chunks_of_X_symm.py`** — coprime counts per fixed-size block of a primorial. (CLI: `--pmax 19 --block 510510`)
- **`Nonlinear_Arithmetic_Sets.py`** — signed-sum sets that cover every coprime mod 210.

### `prime_set_constructions/` — making primes from small-prime sets
- **`combinatorics_lemma.py`** — the "omission-block lemma": partition a set so each
  element is missing from exactly one term, sign each term `±`, and count how many
  signed sums land on primes.
- **`combinatorics_lemma_exponents.py`** — adds factor-level exponents on one active term.
- **`combinatorics_lemma_exponents_exhaustive.py`** — the exhaustive search, recording
  representations of each prime reached.
- **`Clustering_Prime_Products.py`** — branch-and-bound partition of the first k primes
  to make the group products as balanced as possible.

### `prime_patterns/` — primality scans
- **`head_tail.py`** — primality of `A·baseᵏ + X` over `(A, k)` grids vs. the expected
  prime count `Σ 1/log N`. **Generates the heatmaps** in `outputs/heatmaps/`.
- **`Prime_deadends.py`** — DFS that prepends digits to a fixed tail, hunting for
  left-truncatable prime chains. Writes `outputs/prefix_chain_summary.txt`.
- **`Prime_Squares_Distance_Log_terms.py`** — how often `q² − p²` is divisible by 5
  over consecutive primes. *(Needs a `primes.txt` — see notes below.)*

### `ml_and_visuals/` — the side-quests
- **`generateplot.py`** — animates logistic-regression gradient descent; **produces
  `outputs/linear_classifier_training.mp4`**. *(Needs ffmpeg — see notes.)*
- **`mnist_keras_intro.py`** — an MNIST + Keras "first neural net" warm-up.
  *(Incomplete: it sets up the model but never calls `.fit()`.)*
- **`region_plot_3d.py`** — an interactive Plotly 3D plot of a calculus region. *(Needs
  `plotly`.)*

---

## Running

This project uses [uv](https://docs.astral.sh/uv/). To set up and run a script:

```bash
uv sync                                   # create .venv from pyproject.toml / uv.lock
uv run python primorial_wheels/Chunks_of_X_symm.py --pmax 19 --block 510510
```

Or activate the environment and run directly (also works from PyCharm's "Run"):

```bash
.venv/Scripts/python primorial_wheels/PrimorialGapFreq.py --primes 2 3 5 7 --next 11
```

Scripts can be launched from anywhere — each one bootstraps the repo root onto
`sys.path` so it can `import prime_lib`, and writes its artifacts to `outputs/`
regardless of the current working directory. Most scripts have a config block at
the top (constants or `argparse` flags) you can edit.

Requires **Python ≥ 3.13**. Heavy scripts (e.g. `head_tail.py` with large `k`,
`extended_orbits.py` with big primorials) can run for a while — start with the
small example configs.

---

## Notes & caveats

- **`prime_patterns/Prime_Squares_Distance_Log_terms.py`** needs an input file
  `primes.txt` (one prime per line) in the working directory. It is **not** in the
  repo; generate or download a prime list and point `PRIME_FILE` at it.
- **`ml_and_visuals/generateplot.py`** needs an `ffmpeg` binary. Set the
  `FFMPEG_PATH` environment variable or put `ffmpeg` on your `PATH`.
- **`ml_and_visuals/region_plot_3d.py`** needs `plotly` (`pip install plotly`); it is
  not listed in `pyproject.toml`.
- **`ml_and_visuals/mnist_keras_intro.py`** is intentionally incomplete (no training step).
- **`primorial_wheels/Nonlinear_Arithmetic_Sets.py`** starts its search immediately on
  run (no `__main__` guard); edit the `T` at the bottom to control the search size.
- Primality for large integers uses **probabilistic** Miller–Rabin
  (`prime_lib.is_probable_prime`), which is more than reliable enough here but is not a
  certificate of primality.
