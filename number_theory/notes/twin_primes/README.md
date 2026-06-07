# Twin primes — working notes

A line of constructions and the precise point where each one stops. The single
thread: **build an integer (or a pair) with no small prime factors, and hope it
is prime** — see [the single spine](01-the-single-spine.md). Everything reduces to
placing an *admissible center* in a *short window*, and is held shut by two walls
(short-interval existence and parity).

## Notation
For a threshold $L$, let $W_L = \prod_{p\le L} p$ (primorial). A *center* is an
integer $m$ with pair $(m-1,m+1)$. Call $m$ **$L$-admissible** if
$$\gcd\big((m-1)(m+1),\,W_L\big)=1 \iff m\not\equiv\pm1\pmod p \text{ for all } p\le L.$$
The count of admissible residues mod $W_L$ is $\nu(L)=\prod_{2<p\le L}(p-2)$ (the
factor at $2$ is $1$); the density is
$$\frac{\nu(L)}{W_L}=\tfrac12\prod_{2<p\le L}\big(1-\tfrac2p\big)\asymp\frac{1}{(\log L)^2}.$$
(A self-contained warm-up identity, orthogonal to the constructions: for distinct
primes $p,q$, $\;p^{q-1}+q^{p-1}\equiv1\pmod{pq}$; the coprime form is
$a^{\varphi(b)}+b^{\varphi(a)}\equiv1\pmod{ab}$.)

## The notes
1. [The single spine](01-the-single-spine.md) — the one move behind every construction, and the two places it stops.
2. [Centers and the certificate](02-centers-and-the-certificate.md) — admissible centers in $(L,L^2)$ are twin centers; the difference rule.
3. [The lattice obstruction](03-the-lattice-obstruction.md) — congruence control fixes a coset of spacing $W_L=e^{(1+o(1))L}$, dwarfing the $L^2$ window.
4. [Omission / CRT / Fermat toolkit](04-omission-crt-fermat-toolkit.md) — $E_i=M/p_i$, the CRT idempotent basis, and why it builds candidates not centers.
5. [Euclid, and the finite-twins split](05-euclid-and-the-finite-twins-split.md) — why Euclid's modular argument works and this one doesn't.
6. [Coefficient form and the resultant detector](06-coefficient-form-and-resultant-detector.md) — symmetric functions of gaps; $\operatorname{Res}(f(x),f(x+2))$.
7. [The modular ceiling](07-the-modular-ceiling.md) — reduction mod $\ell$ discards gap size; residue data bottoms out at the singular series.
8. [The two walls and parity](08-the-two-walls-and-parity.md) — short-interval existence (A) and the parity barrier (B), stated precisely.
9. [Directions, and what to keep](09-directions-and-what-to-keep.md) — what's worth keeping and where it connects to established results.

## Notes ↔ scripts
| Note | Script(s) |
|---|---|
| 2 — centers / certificate | [`primorials_and_twin_primes/`](../../primorials_and_twin_primes/) (offset lemma); [`prime_patterns/prime_deadends.py`](../../prime_patterns/prime_deadends.py) |
| 3 — first-failure / offsets near a primorial | [`primorials_and_twin_primes/`](../../primorials_and_twin_primes/): `primorial_to_prime_distance_run_ratios.py`, `primorial_radius_scan_twin_prime_factor.py`, `plus_or_minus_primorial_squares.py` |
| 4 — omission / CRT basis | [`prime_set_constructions/`](../../prime_set_constructions/): `combinatorics_lemma*.py` |
| 7 — wheel density / residue counts | [`primorial_wheels/`](../../primorial_wheels/): `chunks_of_x_symm.py`, `chunks_of_6_symm.py`, `orbit_primorials.py`, `extended_orbits.py`, `primorial_gap_freq.py` |
| 8 — maximum-gap (the missing object) | [`primorial_wheels/primorial_gap_density_count.py`](../../primorial_wheels/primorial_gap_density_count.py) |
| covering / card model | [`primorial_wheels/nonlinear_arithmetic_sets.py`](../../primorial_wheels/nonlinear_arithmetic_sets.py) |
| 6 — shift-resultant detector | *(no script yet)* |
