# Research Overview — Modular Sieves, Primes & Primorials

This document ties the scripts in this repository to the ideas behind them. The
unifying subject is **how modular constraints (residues modulo small primes, and
modulo primorials) preserve, split, and remove prime / twin-prime candidates.**

A longer, informal write-up of the underlying ideas is kept locally in
`private/` (git-ignored, not published). This overview is the neutral, public
map: each idea below is stated plainly, with pointers to the code that explores
it and an honest note on what the code does and does not establish.

## Contents
1. [The core picture](#1-the-core-picture)
2. [Crosswalk: idea → script](#2-crosswalk-idea--script)
3. [Theme 1 — Wheel structure & residue counting](#3-theme-1--wheel-structure--residue-counting)
4. [Theme 2 — Primes near a primorial](#4-theme-2--primes-near-a-primorial)
5. [Theme 3 — Building coprimes from small-prime sets](#5-theme-3--building-coprimes-from-small-prime-sets)
6. [Theme 4 — Primality scans & the density question](#6-theme-4--primality-scans--the-density-question)
7. [Theme 5 — Side experiments](#7-theme-5--side-experiments)
8. [Ideas without a script yet](#8-ideas-without-a-script-yet)
9. [Open question / where this points](#9-open-question--where-this-points)

---

## 1. The core picture

Represent an integer `n` by its residue vector

```
v_k(n) = (n mod p_1, n mod p_2, ..., n mod p_k)
```

over the first `k` primes. Then:

- `n` is divisible by one of the first `k` primes  ⟺  some coordinate is `0`.
- `n` is **twin-incompatible**  ⟺  some coordinate is `0` or `2`
  (i.e. `n` or `n−2` is divisible by that prime).

Over a complete cycle modulo the primorial `M_k = p_1·p_2·…·p_k`, these
constraints have **exact** structure: counts, periodicity, and recursive "wheel"
nesting are all computable. A prime gap or a twin-prime drought becomes a
**covering problem** — a run of consecutive residue vectors that all land on a
forbidden coordinate.

The recurring theme across the scripts is the difference between two things:

- **Full-cycle structure** (counts, density, average spacing) — exact and easy.
- **Short-interval behavior** (is there a survivor *here*, before some bound) —
  the genuinely hard part, governed by **maximum-gap** rather than average-gap.

Most scripts here measure or construct the full-cycle structure; the honest open
problem in every thread is the short-interval / maximum-gap control.

---

## 2. Crosswalk: idea → script

| Idea | Script(s) | What the code does |
|------|-----------|--------------------|
| Wheel lifting (each new prime expands the cycle, removes predictable residues) | `primorial_wheels/primorial_gap_freq.py` | Tracks how gaps merge when a wheel is lifted by its next prime |
| Reduced-residue / coprime counts of a primorial | `primorial_wheels/chunks_of_x_symm.py`, `orbit_primorials.py`, `extended_orbits.py` | Counts / reconstructs the integers coprime to a primorial |
| `6x ≡ ±1 (mod p)` residue symmetry | `primorial_wheels/chunks_of_6_symm.py` | Visualizes the `6k ± 1` structure of totatives |
| Maximum-gap / longest covered run (the missing object) | `primorial_wheels/primorial_gap_density_count.py` | Searches worst-case deviation of a model period per residue class |
| Card / residue-covering model | `primorial_wheels/nonlinear_arithmetic_sets.py` | Signed-sum sets covering all coprimes mod 210 |
| Primorial-neighborhood offset lemma | `primorials_and_twin_primes/primorial_to_prime_distance_run_ratios.py` | First composite distance from a primorial |
| Empirical twin-factor data | `primorials_and_twin_primes/plus_or_minus_primorial_squares.py`, `primorial_radius_scan_twin_prime_factor.py` | Scans `p_n# ± offset` for twin-prime factors |
| Midpoint / center-coordinate expressions | `primorials_and_twin_primes/primorial_center_radii_twinp.py` | Exact rational `L/R` expressions around `P/A²` |
| Euclid-like omission sums = CRT basis | `prime_set_constructions/combinatorics_lemma*.py` | "Each element missing from one term," counts primes among signed sums |
| Balanced prime-product partitions | `prime_set_constructions/clustering_prime_products.py` | Branch-and-bound partition of the first `k` primes |
| Density vs. actual prime counts | `prime_patterns/head_tail.py` | Primality of `A·baseᵏ + X` grids vs. expected `Σ 1/log N` |
| Left-truncatable prime chains | `prime_patterns/prime_deadends.py` | Digit-prepending DFS keeping primality |
| Residue statistics over primes | `prime_patterns/prime_squares_distance_log_terms.py` | How often `q² − p²` is divisible by 5 |

---

## 3. Theme 1 — Wheel structure & residue counting
**Folder:** `primorial_wheels/`

The reduced residue system modulo a primorial has clean recursive structure:
adding a prime `p_{k+1}` multiplies the cycle into `p_{k+1}` copies and removes a
predictable set of residues (one for the "prime" wheel, two — `0` and `2` — for
the "twin" wheel). This yields exact counts (Euler-totient products, and
`∏(p_i − 2)` for twin-compatible residues).

- **`primorial_gap_freq.py`** makes the *lifting* step concrete: it sieves a wheel
  by its next prime and records how the bordering gaps merge.
- **`chunks_of_6_symm.py`** shows the `6k ± 1` residue symmetry directly.
- **`chunks_of_x_symm.py`** counts coprimes per fixed-size block — the full-cycle
  density made visible.
- **`orbit_primorials.py` / `extended_orbits.py`** reconstruct the coprime set of
  a primorial from `C ± Pˣ` "orbits" and compare against Euler's φ(N) — an
  explicit handle on the residue-vector viewpoint.
- **`primorial_gap_density_count.py`** is the most important one conceptually: it
  searches for the *worst-case* deviation (longest under-covered run) of a model
  period. That maximum-gap quantity is exactly the object the whole body of work
  keeps needing and not having.

## 4. Theme 2 — Primes near a primorial
**Folder:** `primorials_and_twin_primes/`

**The offset lemma (valid):** if `M = p_k#` and `d` is composite with
`1 < d < p_{k+1}²`, then `d` shares a small prime factor with `M`, so `M ± d`
is composite (barring tiny edge cases). Consequently, a prime appearing within
`p_{k+1}²` of a primorial is *forced* to sit at a **prime** offset — the
composite offsets in that range are structurally killed. The first composite
offset that can escape is `p_{k+1}²` itself.

- **`primorial_to_prime_distance_run_ratios.py`** measures the first composite
  distance from each primorial — a direct probe of that "first failure."
- **`plus_or_minus_primorial_squares.py`** and
  **`primorial_radius_scan_twin_prime_factor.py`** are the empirical
  twin-factor scans: they factor `p_n# ± offset` and look for twin-prime
  factors. These are exploratory data runs — interesting as data, not as a
  theorem (the right way to extend them is hypothesis-first: fix the statistic,
  a comparison set, and a stopping rule before running).
- **`primorial_center_radii_twinp.py`** works with exact expressions around the
  midpoint `m ≈ P/A²` (the center-coordinate / reflection-symmetry idea).

## 5. Theme 3 — Building coprimes from small-prime sets
**Folder:** `prime_set_constructions/`

**Omission sums = CRT basis (valid).** For `M = ∏ p_i`, the omission term
`E_i = M / p_i` is divisible by every prime in the set except `p_i`. A signed
combination `N = Σ c_i E_i` with `p_i ∤ c_i` is therefore coprime to `M` — these
`E_i` are a Chinese-Remainder coordinate basis. The construction makes numbers
*coprime to a fixed set of primes*, which produces **candidates**, not primes
(a larger prime can still divide `N`).

- **`combinatorics_lemma.py` → `combinatorics_lemma_exponents.py` →
  `combinatorics_lemma_exponents_exhaustive.py`** implement exactly this "each
  element missing from one term" construction and count how many of the signed
  sums land on primes. (Concretely, `2·3·7 − 2·3·5 + 2·5·7 + 3·5·7` is one such
  term set for `S = {2,3,5,7}`.)
- **`clustering_prime_products.py`** is a related combinatorial sandbox:
  partition the first `k` primes into groups with balanced products.

## 6. Theme 4 — Primality scans & the density question
**Folder:** `prime_patterns/`

This is the empirical counterpart to the central methodological point: a
full-cycle **density** (e.g. the twin-compatible density `δ_k`) gives the
*average* spacing of survivors, but does **not** by itself guarantee a survivor
in a particular short interval — that needs maximum-gap control. So these
scripts *measure* real counts against the density model instead of assuming it.

- **`head_tail.py`** — primality of `N = A·baseᵏ + X` over `(A, k)` grids,
  compared against the expected count `Σ 1/log N`. This is the "modular tail"
  experiment; the private archive doesn't name it, but it belongs here: it is a
  direct empirical check of *actual prime count vs. density expectation* for a
  structured family of integers. It also produces the heatmaps in `outputs/`.
- **`prime_deadends.py`** — a digit-prepending DFS for left-truncatable prime
  chains (a residue/positional search).
- **`prime_squares_distance_log_terms.py`** — residue statistics of `q² − p²`
  over consecutive primes.

## 7. Theme 5 — Side experiments
**Folder:** `ml_and_visuals/`

Unrelated to the number theory — practice / visualization:
`logistic_regression_animation.py` (logistic-regression animation), `mnist_keras_intro.py` (an
incomplete MNIST/Keras warm-up), `region_plot_3d.py` (a calculus region plot).

---

## 8. Ideas without a script yet

Several ideas in the private archive are not yet represented in code — natural
next experiments if this project grows:

- **Bounded twin-prime sieve certificate** — inside `2 < n < p_{k+1}²`, removing
  residues `0` and `2` modulo every prime `≤ p_k` certifies that `n` and `n−2`
  are both prime. (Example: for primes up to 5, the survivors below `49` are
  `{1, 13, 19, 31, 43}` — the upper members of the twin pairs below 49.) A short
  verifier script would make this concrete.
- **`X + P` local screen** — search `1 ≤ P < X` so that `X+P` and `X+P−2` avoid
  every prime below `X`; a valid certificate, with the open part being whether
  such a `P` exists below `X`.
- **Factorial detector** — `f(n) = (n−1)! / n` is a non-integer exactly when `n`
  is prime or `n = 4` (a Wilson-adjacent observation). A few lines to script.
- **Maximum-run / `J₂(k)` computation** — the longest run with no
  twin-compatible residue modulo `M_k`. `primorial_gap_density_count.py` is the
  closest existing code; a dedicated branch-and-bound over residue vectors would
  target it directly.

## 9. Open question / where this points

Across every thread, the same gap recurs: exact full-cycle structure is in hand,
but turning it into a **short-interval existence** statement needs a bound on the
**maximum gap** between admissible residues — not just their density. The nearest
established mathematics is the **Jacobsthal-function** family and **covering
systems** (and, for the heuristic side, the Hardy–Littlewood prime `k`-tuple
model). That is the most useful direction if this work is ever continued.
