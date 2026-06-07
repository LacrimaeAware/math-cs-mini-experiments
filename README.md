# MathExperimentation

A personal **experimental-math & code playground**. Currently centered on
**number theory** (primes, primorials, twin primes, modular sieves), with room to
grow into other domains. It's a lab notebook in code: each script runs standalone
and prints or plots what it found.

## Map

```
number_theory/                  THE MAIN BODY — number-theory experiments + notes
  README.md                       area hub + a notes <-> scripts crosswalk
  notes/                          written math (start: twin_primes_consolidated.md)
  primorials_and_twin_primes/     primes near a primorial (offset lemma, twin-factor scans)
  primorial_wheels/               coprime / residue / gap structure of primorial moduli
  prime_set_constructions/        omission-sum / CRT-basis constructions
  prime_patterns/                 primality scans (heatmaps, left-truncatable chains)

ml_and_visuals/                 side experiments (logistic-reg animation, MNIST, 3D plot)

docs/research_overview.md       deep script-by-script crosswalk of the number-theory ideas
prime_lib.py                    shared helpers: primality, sieves, primorials, output paths
main.py                         tiny entry stub
outputs/                        generated images / data / caches (a gallery of past runs)
private/                        (git-ignored) raw archive + personal project log
pyproject.toml / uv.lock        dependencies (managed with uv)
```

## Where to start
- **The math, tied to the code** → [`number_theory/README.md`](number_theory/README.md)
- **The latest written notes** → [`number_theory/notes/twin_primes_consolidated.md`](number_theory/notes/twin_primes_consolidated.md)
- **Deep idea ↔ script map** → [`docs/research_overview.md`](docs/research_overview.md)

## Domains
- **Number theory** (`number_theory/`) — the main, ongoing work; self-contained.
- **Machine learning & visuals** (`ml_and_visuals/`) — a few standalone practice
  scripts (a logistic-regression training animation, an incomplete MNIST warm-up,
  a 3D calculus region plot). Kept separate from the number theory.

New domains can be added later as sibling top-level folders.

## Running

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python number_theory/primorial_wheels/chunks_of_x_symm.py --pmax 19 --block 510510
```

Or run directly (also works from PyCharm's "Run"):

```bash
.venv/Scripts/python number_theory/primorial_wheels/primorial_gap_freq.py --primes 2 3 5 7 --next 11
```

Scripts can be launched from anywhere — each finds the repo root automatically to
`import prime_lib`, and writes artifacts to `outputs/`. Most have a config block
(constants or `argparse`) at the top. Requires **Python ≥ 3.13**; some scripts
(large `k`, large primorials) run for a while — start with the small example configs.

## Notes & caveats
- **`number_theory/prime_patterns/prime_squares_distance_log_terms.py`** needs a
  `primes.txt` (one prime per line) in the working directory — not included.
- **`ml_and_visuals/logistic_regression_animation.py`** needs `ffmpeg` (set
  `FFMPEG_PATH` or put it on `PATH`).
- **`ml_and_visuals/region_plot_3d.py`** needs `plotly` (not in `pyproject.toml`).
- **`ml_and_visuals/mnist_keras_intro.py`** is intentionally incomplete (no `.fit()`).
- **`number_theory/primorial_wheels/nonlinear_arithmetic_sets.py`** starts its
  search on run (no `__main__` guard); edit `T` at the bottom.
- Large-integer primality uses probabilistic Miller–Rabin (`prime_lib.is_probable_prime`).

## Publishing
The tracked files are clean for public release (raw / personal notes are
git-ignored in `private/`). Publish steps are in `private/PROJECT_LOG.md`.
