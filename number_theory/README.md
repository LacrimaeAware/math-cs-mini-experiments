# Number Theory

The main body of this project: experiments and written notes on primes,
primorials, twin primes, and modular sieves. Self-contained — the scripts here
import shared helpers from the repo-root `prime_lib.py` (located automatically, so
they run from anywhere).

## Code
```
primorials_and_twin_primes/     primes in the neighborhood of a primorial
primorial_wheels/               coprime / residue / gap structure of primorial moduli
prime_set_constructions/        omission-sum / CRT-basis constructions
prime_patterns/                 primality scans (heatmaps, left-truncatable chains)
```
A script-by-script crosswalk (each idea ↔ the code) is in
[`../docs/research_overview.md`](../docs/research_overview.md).

## Notes
The written math lives in [`notes/`](notes/), split into short topic files (one
idea each) rather than one long document:

- [`notes/twin_primes/`](notes/twin_primes/) — the twin-prime line: centers and
  the certificate, the lattice obstruction, the omission/CRT toolkit, the
  Euclid comparison, the coefficient/resultant detector, the modular ceiling, the
  two walls (short-interval existence + parity), and the directions worth pursuing.
  Start at its [index](notes/twin_primes/README.md), which also maps each note to
  the scripts.
