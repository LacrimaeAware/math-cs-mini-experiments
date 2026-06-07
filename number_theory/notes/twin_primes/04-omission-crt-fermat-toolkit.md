# Omission / CRT / Fermat toolkit

Let $S=\{p_1,\dots,p_k\}$, $M=\prod p_i$, $E_i=M/p_i$. The normalized idempotents
$B_i=E_i(E_i^{-1}\bmod p_i)$ satisfy $B_i\equiv\delta_{ij}\pmod{p_j}$, so
$\sum a_iB_i$ realizes any residue vector mod $M$, the CRT idempotent basis. With
free coefficients it represents every class mod $M$.

The Fermat layer: for a prime $r\notin S$, each $E_i$ is coprime to $r$, so
$E_i^{r-1}\equiv1\pmod r$, and $N=\sum_i\sigma_iE_i^{r-1}\equiv\sum_i\sigma_i\pmod r$.
With a two-block split $A,B$ of $S$, $A^{r-1}+B^{r-1}\equiv2\pmod r$ while $N$ is
coprime to all of $S$. **Correction:** the exponent is $r-1$ because $r$ is
*prime*; for composite modulus $n$ the right exponent is a multiple of the
Carmichael function $\lambda(n)$ (Euler's $\varphi(n)$ suffices), not $n-1$.
"$a^{n-1}\equiv1$ for all coprime $a$" with composite $n$ is the accident defining
Carmichael numbers. The thing that need only be coprime is the *base*, not the
modulus.

What this controls is $N$ mod small primes: it makes $N$ itself coprime to
$S\cup\{r\}$, a residue-controlled **Euclid candidate** (a single number, no small
factors), *not* a center. It says nothing about $N\pm1$. Upgrading to a twin
generator means controlling $N\pm1$ simultaneously, i.e. demanding
$N\not\equiv\pm1$ at every small prime, which returns to
[centers and the certificate](02-centers-and-the-certificate.md) and inherits
[the lattice obstruction](03-the-lattice-obstruction.md). Its productive home is
the *mirror* problem: **covering systems** (Sierpiński/Riesel numbers, Erdős
congruences), where this exact residue control proves compositeness. Twin primes
asks for the *absence* of such a covering.
