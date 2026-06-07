# Number theory

Number-theory scripts and notes on primes, primorials, twin primes, and modular
sieves. Scripts import shared helpers from the repository-root `prime_lib.py` and
locate it automatically, so they run from any directory.

## Code

```text
primorials_and_twin_primes/   primes in the neighborhood of a primorial
primorial_wheels/             coprime, residue, and gap structure of primorial moduli
prime_set_constructions/      omission-term and CRT-basis constructions
prime_patterns/               primality scans (heatmaps, left-truncatable chains)
```

A per-script description of the ideas is in
[../docs/research_overview.md](../docs/research_overview.md).

## Notes

Written notes are in [`notes/`](notes/), split into short topic files.

- [`notes/twin_primes/`](notes/twin_primes/): the twin-prime constructions and the
  point where each one stops (centers and the certificate, the lattice
  obstruction, the omission/CRT toolkit, the Euclid comparison, the
  coefficient/resultant detector, the modular ceiling, the two walls, and
  directions). The [index](notes/twin_primes/README.md) maps each note to the
  scripts.
