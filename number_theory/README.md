# Number Theory

The main body of this project: experiments and written notes on primes,
primorials, twin primes, and modular sieves. Self-contained — the scripts here
import shared helpers from the repo-root `prime_lib.py` (they locate it
automatically, so they run from anywhere).

## What's in here
```
notes/                          written math — the ideas, formalized
  twin_primes_consolidated.md     the current consolidated write-up (read this first)
primorials_and_twin_primes/     primes in the neighborhood of a primorial
primorial_wheels/               coprime / residue / gap structure of primorial moduli
prime_set_constructions/        omission-sum / CRT-basis constructions
prime_patterns/                 primality scans (heatmaps, left-truncatable chains)
```

`notes/twin_primes_consolidated.md` is the latest, sharpest summary of the whole
line of thinking: centers/admissibility and the prime-square certificate, the
`X+P` construction, the omission/CRT-idempotent toolkit, the coefficient/
resultant twin detector, the "modular ceiling" (why residues bottom out at the
singular series), and the two walls — short-interval existence and parity.

A deeper, script-by-script crosswalk (each idea ↔ the code that explores it) is
in [`../docs/research_overview.md`](../docs/research_overview.md).

## Notes ↔ scripts
The consolidated notes and the code are two views of the same ideas:

| Notes section | Idea | Script(s) |
|---|---|---|
| §2 centers / admissibility & certificate | an admissible center in `(L, L²)` certifies a twin pair | `primorials_and_twin_primes/` (offset lemma); `prime_patterns/prime_deadends.py` |
| §4 lattice obstruction / first-failure | composite offsets near a primorial are killed below `p_{k+1}²` | `primorials_and_twin_primes/primorial_to_prime_distance_run_ratios.py`, `primorial_radius_scan_twin_prime_factor.py`, `plus_or_minus_primorial_squares.py` |
| §5 omission / CRT idempotent basis | `E_i = M/p_i`; signed sums land coprime to the primorial | `prime_set_constructions/combinatorics_lemma*.py` |
| §1 & §10 wheel density / residue counts | `(1/2)∏(p−2)/p`; residue-count data = the singular series | `primorial_wheels/chunks_of_x_symm.py`, `chunks_of_6_symm.py`, `orbit_primorials.py`, `extended_orbits.py`, `primorial_gap_freq.py` |
| §11(A) maximum-gap (the missing object) | average gap ≠ max gap; need short-interval control | `primorial_wheels/primorial_gap_density_count.py` |
| covering / card model | cover the coprimes mod 210 with chosen residue classes | `primorial_wheels/nonlinear_arithmetic_sets.py` |
| §9 shift-resultant detector | `Res(f(x), f(x+2)) = ∏ f(q_i+2)` | *(no script yet — see notes §12 and overview §8)* |

## Fuller archive (lives elsewhere)
This repo holds the code plus the latest consolidated write-up. The complete
idea-trail, chronology, and raw source notes live in the **`master-organizer`**
vault (a separate repo):
- canonical, synthesized: `master-organizer/vault/10_canonical/projects/research/twin-prime/` (16 notes) and `.../primordial-idiot/`
- raw originals: `master-organizer/vault/00_inbox/drop/Personal Projects n Research/`
