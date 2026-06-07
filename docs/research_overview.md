# Research overview: modular sieves, primes, and primorials

Per-script description of the number-theory ideas in this repository. The common
subject is how modular constraints (residues modulo small primes and modulo
primorials) preserve, split, and remove prime and twin-prime candidates. Paths
below are under `number_theory/`.

## Core picture

An integer `n` is represented by its residue vector over the first `k` primes:

```
v_k(n) = (n mod p_1, n mod p_2, ..., n mod p_k)
```

`n` is divisible by one of the first `k` primes iff some coordinate is 0. `n` is
twin-incompatible iff some coordinate is 0 or 2 (`n` or `n-2` is divisible by that
prime). Over a complete cycle modulo the primorial `M_k = p_1*p_2*...*p_k`, these
constraints have exact structure: counts, periodicity, and recursive wheel nesting
are computable. A prime gap or a twin-prime drought is then a covering problem: a
run of consecutive residue vectors that all land on a forbidden coordinate.

The recurring distinction is between full-cycle structure (counts, density,
average spacing), which is exact and computable, and short-interval behavior
(whether a survivor exists before a given bound), which is governed by the maximum
gap rather than the average gap. Most scripts measure or construct the full-cycle
structure. The open part in each thread is short-interval (maximum-gap) control.

## Idea to script

| Idea | Script | What it computes |
|---|---|---|
| Wheel lifting (a new prime expands the cycle and removes fixed residues) | `primorial_wheels/primorial_gap_freq.py` | How bordering gaps merge when a wheel is sieved by its next prime |
| Coprime / reduced-residue counts of a primorial | `primorial_wheels/chunks_of_x_symm.py`, `orbit_primorials.py`, `extended_orbits.py` | Counts and reconstructs the integers coprime to a primorial |
| `6x = +/-1 (mod p)` residue symmetry | `primorial_wheels/chunks_of_6_symm.py` | The `6k +/- 1` structure of totatives |
| Maximum gap / longest covered run | `primorial_wheels/primorial_gap_density_count.py` | Worst-case deviation of a model period per residue class |
| Residue-covering (card) model | `primorial_wheels/nonlinear_arithmetic_sets.py` | Signed-sum sets covering the coprimes mod 210 |
| Primorial-neighborhood offset lemma | `primorials_and_twin_primes/primorial_to_prime_distance_run_ratios.py` | First composite distance from a primorial |
| Twin-prime-factor scans | `primorials_and_twin_primes/plus_or_minus_primorial_squares.py`, `primorial_radius_scan_twin_prime_factor.py` | Factors `p_n# +/- offset`; tests for twin-prime factors |
| Center-coordinate expressions | `primorials_and_twin_primes/primorial_center_radii_twinp.py` | Exact rational expressions around `P/A^2` |
| Omission sums = CRT basis | `prime_set_constructions/combinatorics_lemma*.py` | Counts primes among signed sums of omission terms |
| Balanced prime-product partitions | `prime_set_constructions/clustering_prime_products.py` | Branch-and-bound partition of the first `k` primes |
| Density vs. measured prime counts | `prime_patterns/head_tail.py` | Primality of `A*base^k + X` grids vs. expected `sum 1/log N` |
| Left-truncatable prime chains | `prime_patterns/prime_deadends.py` | Digit-prepending search keeping primality |
| Residue statistics over primes | `prime_patterns/prime_squares_distance_log_terms.py` | Fraction of `q^2 - p^2` divisible by 5 |

## Wheel structure and residue counting

Folder: `primorial_wheels/`. The reduced residue system modulo a primorial has
recursive structure: adding a prime `p_{k+1}` multiplies the cycle into `p_{k+1}`
copies and removes a fixed set of residues (one for the coprime wheel; two, 0 and
2, for the twin wheel). This gives exact counts: Euler-totient products, and
`prod(p_i - 2)` over odd primes for twin-compatible residues.

- `primorial_gap_freq.py` sieves a wheel by its next prime and records how the
  bordering gaps merge (the lifting step).
- `chunks_of_6_symm.py` prints the `6k +/- 1` residue structure.
- `chunks_of_x_symm.py` counts coprimes per fixed-size block (full-cycle density).
- `orbit_primorials.py` and `extended_orbits.py` reconstruct the coprime set of a
  primorial from `C +/- P^x` orbits and compare against `phi(N)`.
- `primorial_gap_density_count.py` searches the worst-case deviation (longest
  under-covered run) of a model period. This maximum-gap quantity is the object the
  short-interval question requires.

## Primes near a primorial

Folder: `primorials_and_twin_primes/`. Offset lemma: if `M = p_k#` and `d` is
composite with `1 < d < p_{k+1}^2`, then `d` shares a small prime factor with `M`,
so `M +/- d` is composite (apart from edge cases). A prime within `p_{k+1}^2` of a
primorial therefore sits at a prime offset; composite offsets in that range are
removed. The first composite offset that can escape is `p_{k+1}^2`.

- `primorial_to_prime_distance_run_ratios.py` measures the first composite distance
  from each primorial.
- `plus_or_minus_primorial_squares.py` and
  `primorial_radius_scan_twin_prime_factor.py` factor `p_n# +/- offset` and test
  for twin-prime factors. These are data scans; a controlled version fixes the
  statistic, a comparison set, and a stopping rule before running.
- `primorial_center_radii_twinp.py` evaluates exact expressions around the midpoint
  `m = nearest_int(P/A^2)`.

## Building coprimes from small-prime sets

Folder: `prime_set_constructions/`. Omission sums as a CRT basis: for
`M = prod p_i`, the omission term `E_i = M / p_i` is divisible by every prime in
the set except `p_i`. A signed combination `N = sum c_i E_i` with `p_i` not
dividing `c_i` is coprime to `M`. The construction yields numbers coprime to a
fixed set of primes (candidates), not primes; a larger prime can still divide `N`.

- `combinatorics_lemma.py`, `combinatorics_lemma_exponents.py`, and
  `combinatorics_lemma_exponents_exhaustive.py` implement the "each element missing
  from one term" construction and count how many signed sums are prime. Example
  term set for `S = {2,3,5,7}`: `2*3*7 - 2*3*5 + 2*5*7 + 3*5*7`.
- `clustering_prime_products.py` partitions the first `k` primes into groups with
  balanced products.

## Primality scans

Folder: `prime_patterns/`. A full-cycle density (for twins, `delta_k`) gives the
average spacing of survivors but does not guarantee a survivor in a given short
interval; that requires maximum-gap control. These scripts measure counts against
the density model.

- `head_tail.py` tests primality of `N = A*base^k + X` over `(A, k)` grids and
  compares the count to the expected `sum 1/log N`. Writes the heatmaps in
  `outputs/`.
- `prime_deadends.py` runs a digit-prepending search for left-truncatable prime
  chains.
- `prime_squares_distance_log_terms.py` reports the fraction of `q^2 - p^2`
  divisible by 5 over consecutive primes.

## Side scripts

Folder: `ml_and_visuals/`. Not number theory:
`logistic_regression_animation.py` (a logistic-regression training animation),
`mnist_keras_intro.py` (MNIST with a dense Keras model, no training step),
`region_plot_3d.py` (a 3D region plot).

## Ideas without a script

- Bounded twin-prime certificate: inside `2 < n < p_{k+1}^2`, removing residues 0
  and 2 modulo every prime `<= p_k` certifies that `n` and `n-2` are both prime.
  For primes up to 5 the survivors below 49 are `{1, 13, 19, 31, 43}`.
- `X + P` screen: search `1 <= P < X` so that `X+P` and `X+P-2` avoid every prime
  below `X`.
- Factorial detector: `f(n) = (n-1)! / n` is a non-integer exactly when `n` is
  prime or `n = 4`.
- Maximum-run `J_2(k)`: the longest run with no twin-compatible residue modulo
  `M_k`. `primorial_gap_density_count.py` is the closest existing code.

The twin-prime notes in `number_theory/notes/twin_primes/` develop these.

## Where this points

Each thread has exact full-cycle structure but needs a bound on the maximum gap
between admissible residues, not just their density, to state short-interval
existence. The nearest established mathematics is the Jacobsthal function and
covering systems, with the Hardy-Littlewood prime k-tuple model for the heuristic
side.
