# The two structural walls

**(A) Short-interval existence.** Constructions pin a center to a coset of
$W_L\mathbb Z$ with $W_L=e^{(1+o(1))L}$, while certification needs a representative
in a window of length $\sim L^2$. The expected count there is large (see
[the lattice obstruction](03-the-lattice-obstruction.md)); turning "expected" into
"guaranteed without an equidistribution hypothesis" is the open problem.

**(B) Parity, stated precisely.** Two different things usually get merged. The
parity *phenomenon* is **proven**: Selberg, then Bombieri, gave explicit sequences
with identical sieve data (sifting density plus distribution in progressions up to
modulus $\lesssim x^{1/2}$) where one lives on numbers with an even number of prime
factors and one on odd. So that data cannot separate "prime" (one factor) from
"semiprime" (two). The *claim* "therefore twin primes cannot be done by residue
methods" is **not** a theorem, it is an extrapolation, true for a given
construction only insofar as that construction uses only such data; and the barrier
*has been broken* (Friedlander-Iwaniec, primes $a^2+b^4$, by injecting bilinear
information). Operational test: parity blocks a method exactly when its only input
is which residue classes are forbidden modulo which primes (linear / "Type I");
beating it requires bilinear ("Type II") information, control of $\sum_{m,n}a_mb_n$
over products $mn$. Pure CRT, the omission basis, admissibility, centers, and a
single product $\prod_A\cdot\prod_B$ are all Type I, they supply exactly the
information the proven examples show is too weak. What such methods *do* achieve is
**bounded gaps**: $\le70{,}000{,}000$ (Zhang, 2013), $\le246$ (Polymath8b, 2014),
$\le6$ under the generalized Elliott-Halberstam conjecture. Gap exactly $2$ is open
precisely because of parity.
