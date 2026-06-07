# The lattice obstruction

Sharpest form: fix $\{p\le L\}$, take a reference center $C_0\equiv0\pmod{W_L}$,
build centers $C_i$ each forced divisible by $p_i$ (so $p_i$ obstructs no
difference touching $C_i$), and try to drive some center or difference admissible
mod all of $W_L$ while landing in a short window.

> **Lattice obstruction.** Every constraint — "$C_i$ admissible at $p$",
> "$p_i\mid C_i$", "$C_j-C_i$ admissible at $p$" — is a congruence modulo a divisor
> of $W_L$. So each constructed quantity is fixed only modulo $W_L$: its solutions
> are cosets of $W_L\mathbb Z$, spaced $W_L$ apart. Affine combinations and
> differences stay congruences mod $W_L$ and do not shrink the spacing.

By Chebyshev/PNT, $\log W_L=\sum_{p\le L}\log p\sim L$, so $W_L=e^{(1+o(1))L}$,
while the certificate window has length $\sim L^2=e^{2\log L}$. The spacing $e^L$
dwarfs the window $L^2$. Controlling residues selects the coset, not a short
representative; no differencing or forced divisibility narrows the modulus below
$W_L$. This separates two things the construction tends to merge:

- **Existence (favorable).** Heuristic count of admissible centers in $(L,L^2)$ is
  $\sim \mathfrak S\,L^2/(\log L)^2\to\infty$ — the Hardy–Littlewood prediction
  that twin centers are abundant there.
- **Construction (blocked).** Building one by congruences lands it on a lattice of
  spacing $W_L$ and cannot place it in the window.

(Dirichlet: for $\gcd(r,M)=1$ there are infinitely many primes $\equiv r\pmod M$ —
one prime per class, never a constrained *pair*. That missing simultaneity is the
twin condition.)

This is wall (A) in [the two walls and parity](08-the-two-walls-and-parity.md).
