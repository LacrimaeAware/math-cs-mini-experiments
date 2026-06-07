# Twin primes — consolidated working notes

These notes pull together the constructions developed across the source files (the centers/admissibility line, the $X+P$ construction, and the omission/Fermat residue line) and the threads worked out in discussion (the Euclid comparison, the finite twin-free set, the product/coefficient form, the shift-resultant detector, and the modular ceiling). They formalize each idea, mark where each one stops advancing — with the relevant mathematics rather than commentary — and end with the directions that connect this circle of ideas to established results.

## 0. The single spine

Every construction below is one move in different clothing: **build an integer (or a pair) with no small prime factors, and hope it is prime.** They all stop at one of two places:

1. **Short-interval existence** — controlling residues pins a number to a coset of a huge modulus, not to a short window where "no small factors" certifies primality.
2. **The gap from coprime to prime** — "no small factors" means *all* prime factors are large; "prime" means *exactly one* prime factor. Congruence data controls the first and is blind to the second.

A third, unifying observation ties the modular attempts together: the entire *modular* content of the problem is the **singular series** (the twin-prime constant) — the part that is completely understood — and the difficulty lives in the complementary, **non-modular (archimedean)** part.

## 1. Notation

For a threshold $L$, let $W_L = \prod_{p\le L} p$ (primorial). A *center* is an integer $m$ with associated pair $(m-1,m+1)$. Call $m$ **$L$-admissible** if
$$\gcd\big((m-1)(m+1),\,W_L\big)=1 \iff m\not\equiv\pm1\pmod p \text{ for all } p\le L.$$
The count of admissible residues mod $W_L$ is $\nu(L)=\prod_{2<p\le L}(p-2)$ (the factor at $2$ is $1$: admissibility there is just "$m$ even"); the density is
$$\frac{\nu(L)}{W_L}=\tfrac12\prod_{2<p\le L}\big(1-\tfrac2p\big)\asymp\frac{1}{(\log L)^2}.$$

A warm-up identity (correct, self-contained, but orthogonal to the constructions): for distinct primes $p,q$, $\;p^{q-1}+q^{p-1}\equiv1\pmod{pq}$ by Fermat mod each prime plus CRT; the coprime-composite form is $a^{\varphi(b)}+b^{\varphi(a)}\equiv1\pmod{ab}$, and the minus version is mixed-sign ($\equiv-1\pmod a$, $\equiv+1\pmod b$).

## 2. Centers/admissibility and the certificate

> **Certificate.** Let $L\ge2$ and $m$ be $L$-admissible with $L<m-1$ and $m+1\le L^2$. Then $m-1$ and $m+1$ are both prime.

*Proof.* Admissibility ⟹ $m\pm1$ have no prime factor $\le L$. A composite $n$ has a prime factor $\le\sqrt n$; $m\pm1\le L^2$ gives $\sqrt{m\pm1}<L$, so neither is composite. $\square$

So admissible centers in the window $(L,\,L^2)$ are exactly twin-prime centers (the lower bound $m-1>L$ omits small pairs where a member equals a sieve prime). This is the same engine as the **$X+P$ construction** under the dictionary $X+P-2,\,X+P \leftrightarrow$ center $m=X+P-1$: forbidding $P\equiv-X,-X+2\pmod Q$ for all $Q<X$ is exactly $m\not\equiv\pm1\pmod Q$. The center coordinate makes the symmetry $m\mapsto(m-1,m+1)$ and the link to the singular series explicit.

## 3. The difference rule

> Let $X,Y$ be $L$-admissible, $D=Y-X$, and $q\le L$ prime. If $q\mid X$ or $q\mid Y$, then $D\not\equiv\pm1\pmod q$. If $q\nmid X$ and $q\nmid Y$, then $q$ obstructs $D$ exactly when $Y\equiv X\pm1\pmod q$.

*Proof.* If $q\mid X$ then $D\equiv Y\not\equiv\pm1$; $\{\pm1\}$ is closed under negation, so $q\mid Y$ is identical. Otherwise $D\equiv\pm1$ is the literal obstruction. $\square$

Two corrections fall out as mathematics: "divides neither center ⟹ $q\nmid D$" is false ($X\equiv Y$ gives $q\mid D$), but harmless, since the obstructed residues are $\pm1$ and $D\equiv0$ is not among them. The same fact makes the reference choice $A\equiv0\pmod{W_L}$ clean: $m\mapsto m-A\equiv m\pmod{W_L}$ preserves admissibility.

## 4. The inductive network of centers, and the lattice obstruction

Sharpest form: fix $\{p\le L\}$, take a reference center $C_0\equiv0\pmod{W_L}$, build centers $C_i$ each forced divisible by $p_i$ (so $p_i$ obstructs no difference touching $C_i$), and try to drive some center or difference admissible mod all of $W_L$ while landing in a short window.

> **Lattice obstruction.** Every constraint — "$C_i$ admissible at $p$", "$p_i\mid C_i$", "$C_j-C_i$ admissible at $p$" — is a congruence modulo a divisor of $W_L$. So each constructed quantity is fixed only modulo $W_L$: its solutions are cosets of $W_L\mathbb Z$, spaced $W_L$ apart. Affine combinations and differences stay congruences mod $W_L$ and do not shrink the spacing.

By Chebyshev/PNT, $\log W_L=\sum_{p\le L}\log p\sim L$, so $W_L=e^{(1+o(1))L}$, while the certificate window has length $\sim L^2=e^{2\log L}$. The spacing $e^L$ dwarfs the window $L^2$. Controlling residues selects the coset, not a short representative; no differencing or forced divisibility narrows the modulus below $W_L$. This separates two things the construction tends to merge:

- **Existence (favorable).** Heuristic count of admissible centers in $(L,L^2)$ is $\sim \mathfrak S\,L^2/(\log L)^2\to\infty$ — the Hardy–Littlewood prediction that twin centers are abundant there.
- **Construction (blocked).** Building one by congruences lands it on a lattice of spacing $W_L$ and cannot place it in the window.

(Dirichlet: for $\gcd(r,M)=1$ there are infinitely many primes $\equiv r\pmod M$ — one prime per class, never a constrained *pair*. That missing simultaneity is the twin condition.)

## 5. The omission/Fermat residue toolkit

Let $S=\{p_1,\dots,p_k\}$, $M=\prod p_i$, $E_i=M/p_i$. The normalized idempotents $B_i=E_i(E_i^{-1}\bmod p_i)$ satisfy $B_i\equiv\delta_{ij}\pmod{p_j}$, so $\sum a_iB_i$ realizes any residue vector mod $M$ — the CRT idempotent basis. With free coefficients it represents every class mod $M$.

The Fermat layer: for a prime $r\notin S$, each $E_i$ is coprime to $r$, so $E_i^{r-1}\equiv1\pmod r$, and $N=\sum_i\sigma_iE_i^{r-1}\equiv\sum_i\sigma_i\pmod r$. With a two-block split $A,B$ of $S$, $A^{r-1}+B^{r-1}\equiv2\pmod r$ while $N$ is coprime to all of $S$. **Correction:** the exponent is $r-1$ because $r$ is *prime*; for composite modulus $n$ the right exponent is a multiple of the Carmichael function $\lambda(n)$ (Euler's $\varphi(n)$ suffices), not $n-1$. "$a^{n-1}\equiv1$ for all coprime $a$" with composite $n$ is the accident defining Carmichael numbers. The thing that need only be coprime is the *base*, not the modulus.

What this controls is $N$ mod small primes: it makes $N$ itself coprime to $S\cup\{r\}$ — a residue-controlled **Euclid candidate** (a single number, no small factors), *not* a center. It says nothing about $N\pm1$. Upgrading to a twin generator means controlling $N\pm1$ simultaneously, i.e. demanding $N\not\equiv\pm1$ at every small prime — which returns to §2 and inherits §4's wall. Its productive home is the *mirror* problem: **covering systems** (Sierpiński/Riesel numbers, Erdős congruences), where this exact residue control proves compositeness. Twin primes asks for the *absence* of such a covering.

## 6. Why Euclid's modular argument succeeds where this one stalls

Euclid uses modular reasoning and works — the point is *what he asks of it*. He forms $N=p_1\cdots p_n+1$, concludes $N$ is coprime to every $p_i$, and stops. He never claims $N$ is prime, and it often isn't: $2\cdot3\cdot5\cdot7\cdot11\cdot13+1=30031=59\times509$. All he needs is "$N$ has *some* prime factor, which must be new," and every integer $>1$ has one. His conclusion does not care whether $N$ has one prime factor or twelve.

Twin primes need $m-1$ and $m+1$ each to have *exactly one* prime factor. Coprimality buys "$m\pm1$ have only large factors," but a number with only large factors can still be a product of two large primes. So what coprimality hands Euclid for free — "divisible by something new," true regardless of factor count — is not what twins need — "exactly one prime factor," a statement about the *number* of factors, which congruence data does not encode. That is the precise content behind "coprime $\ne$ prime," and it is about the *kind of conclusion* (factor-count-blind vs factor-count-sensitive), not about bounds. ("Parity" is shorthand for that factor count mod $2$; the claim is that residue information is blind to it.)

## 7. The finite-twins assumption and the A/B split

Assume finitely many twins; split the primes $P=A\sqcup B$ with $B$ the twins, $A$ the rest, and form e.g. $N=\prod_A+\prod_B$. Mod any $q\in P$: if $q\in A$ then $N\equiv\prod_B\not\equiv0$, symmetrically for $q\in B$, so $N$ is coprime to all of $P$ — it has a prime factor outside $P$. This is Euclid with labels, and it leaks: reducing $N$ mod $q$ keeps the residue and discards the label — it has no idea whether $q$ was a twin. The labeling can steer which residue class $N$ lands in, but no choice reaches into the primality of $N$'s *neighbors*, and a twin pair is a fact about neighbors. The output is a new prime, not a new twin.

Under the "$P$ = all primes up to the last twin $T_{\max}$" version, that is not even a contradiction: a prime beyond $T_{\max}$ is guaranteed (primes are infinite whether or not twins are), and the assumption restricts twins, not primes. The whole remaining distance is "new prime ⟹ new twin," exactly the step finiteness gives no grip on; so the bounded-range version affords the same single fact as the arbitrary-range one. The sharpest honest form of "twins are finite" is: *every admissible center $m>T_{\max}$ has a composite (large-factored) neighbor* — every abundant twin-coprime center is spoiled. But spoiling is a primality fact, not chosen by residues, so no modular contradiction follows however tightly the range is bounded.

The trivial deduction "no twins ⟹ consecutive gaps $\ge4$" is true and nearly empty: the average gap near $x$ is $\sim\log x$, which dwarfs $4$ long before anything interesting.

## 8. The product / coefficient form

Write each prime via base-plus-gap and take the product. For $\{2,23,29\}$ (gaps $0,21,27$ from base $2$):
$$p\,(p+21)(p+27)=p^3+48\,p^2+567\,p,\qquad 48=21+27,\;\;567=21\cdot27,$$
checking at $p=2$: $8+192+1134=1334=2\cdot23\cdot29$. The coefficients are the **elementary symmetric functions of the gaps**: each is a power of $p$ times a product of some subset of the gaps. The base prime is "excluded" — its gap is $0$, so it only ever contributes a power of $p$ and never enters a cross-term (which is also the "$p$ appears twice" feeling: $p$ is both the pulled-out factor and the additive piece inside each binomial). Note two distinct objects: gaps-as-offsets gives symmetric functions of the *distances*; adding the base to each prime, $\prod_i(2+q_i)$, gives $2^k$ times subset-products of the *primes*.

## 9. Detecting twins algebraically: shift-and-compare

Let $f(x)=\prod_i(x-q_i)$ (primes as roots; same data as the product above).

> twin-free $\iff$ none of $q_i+2$ is in the set $\iff f(q_i+2)\ne0$ for all $i$.

Picture: lay the set down and a copy shifted right by $2$ on top; a twin pair is an overlap point. Examples: $\{5,7\}$ gives $f(7)=0$ (caught); $\{5,11\}$ gives $f(7)=-8,\,f(13)=16$ (no overlap). The number packaging this is the **resultant** $\operatorname{Res}\!\big(f(x),f(x+2)\big)=\prod_i f(q_i+2)$, zero iff a twin pair exists, and a polynomial in the coefficients of $f$.

Caveat on what this *is*: written as $\prod_i f(q_i+2)$ it is the inert "factor-testing" form — an exhaustive pairwise scan wearing a product sign, equivalent to the very condition it tests, hence a reformulation rather than a lever. But the *instinct* is correct and is exactly how the problem is posed analytically: the twin count is the autocorrelation $\sum_n\Lambda(n)\Lambda(n+2)$, the prime indicator against its own shift by $2$. The resultant is the finite, exact shadow of that sum.

## 10. The modular ceiling

Work the coefficient form mod a small prime $\ell$. With $F(x)=\prod_i(x+a_i)=\sum_k\sigma_k x^{n-k}$,
$$F(x)\equiv\prod_{r\bmod\ell}(x+r)^{m_r}\pmod{\ell},\qquad m_r=\#\{i:a_i\equiv r\}.$$
So every coefficient mod $\ell$ is just bookkeeping of the residue counts $\{m_r\}$ — the distribution of the numbers across classes mod $\ell$ is the *entire* modular content. This cannot see twinness: a real twin differs by *exactly $2$ as integers*, but mod $\ell$ that becomes "two residues differing by $2$ mod $\ell$," a weaker thing. Mod $5$, the twins $11,13$ sit at $1,3$ and the non-twins $7,19$ sit at $2,4$ — both "differ by $2$ mod $5$." Once a set covers more than half the residues mod $\ell$ (any large set does), some pair differs by $2$ mod $\ell$, so $\operatorname{Res}\equiv0\pmod\ell$ for essentially every large set, twin-free or not. The twin/twin-free signal survives only in the *exact integer value*, never in any reduction mod $\ell$.

So the modular route bottoms out not by being "vapid by definition," but because reduction mod small primes is precisely the operation that discards gap *size*: differing-by-$2$ is an archimedean fact, and mod $\ell$ keeps only differing-by-$2$-mod-$\ell$. Assembled over all small $\ell$, the residue-count data is exactly the **singular series** — the twin-prime constant $C_2=\prod_{p>2}\!\big(1-\tfrac1{(p-1)^2}\big)$ — which converges, is computable, and is the local density of twin-admissible residues. Hardy–Littlewood: the count is *that modular constant* times an archimedean term $\sim N/(\log N)^2$. Pushing coefficient/modular algebra harder walks straight into $C_2$ (the solved half); the difficulty sits, provably, in the complementary non-modular term.

## 11. The two structural walls

**(A) Short-interval existence.** Constructions pin a center to a coset of $W_L\mathbb Z$ with $W_L=e^{(1+o(1))L}$, while certification needs a representative in a window of length $\sim L^2$. The expected count there is large (§4); turning "expected" into "guaranteed without an equidistribution hypothesis" is the open problem.

**(B) Parity — stated precisely.** Two different things usually get merged. The parity *phenomenon* is **proven**: Selberg, then Bombieri, gave explicit sequences with identical sieve data (sifting density plus distribution in progressions up to modulus $\lesssim x^{1/2}$) where one lives on numbers with an even number of prime factors and one on odd. So that data cannot separate "prime" (one factor) from "semiprime" (two). The *claim* "therefore twin primes cannot be done by residue methods" is **not** a theorem — it is an extrapolation, true for a given construction only insofar as that construction uses only such data; and the barrier *has been broken* (Friedlander–Iwaniec, primes $a^2+b^4$, by injecting bilinear information). Operational test: parity blocks a method exactly when its only input is which residue classes are forbidden modulo which primes (linear / "Type I"); beating it requires bilinear ("Type II") information, control of $\sum_{m,n}a_mb_n$ over products $mn$. Pure CRT, the omission basis, admissibility, centers, and a single product $\prod_A\cdot\prod_B$ are all Type I — they supply exactly the information the proven examples show is too weak. What such methods *do* achieve is **bounded gaps**: $\le70{,}000{,}000$ (Zhang, 2013), $\le246$ (Polymath8b, 2014), $\le6$ under the generalized Elliott–Halberstam conjecture. Gap exactly $2$ is open precisely because of parity.

## 12. What to keep, and where it leads

Worth keeping, tightened: the **center/admissibility formalism and certificate** (§2); the **difference rule** (§3); the **CRT-idempotent/Fermat toolkit** (§5), relabelled honestly as residue control; the **shift-resultant detector** (§9), understood as the finite shadow of the autocorrelation.

Directions, roughly by payoff:

1. **Admissible $k$-tuples and the singular series.** Centers are the $k=2$ case (offsets $\{0,2\}$). The density $\prod(1-2/p)$ is the $k=2$ singular series; the general theory (Hardy–Littlewood) is where "perfect residue distribution" becomes precise: equidistribution of admissible residues in short windows.
2. **The Maynard–Tao sieve.** The rigorous form of "an admissible pattern eventually contains several primes" — it is the *same* residue/sieve world built carefully and fed a distribution input. It yields bounded gaps but not $2$; seeing why is the efficient way to internalize wall (B).
3. **The parity literature** (Selberg, Bombieri; Friedlander–Iwaniec for the breakthrough) — the ceiling of the approach and the one known way past it.
4. **Covering systems** (Sierpiński/Riesel, Erdős) — the natural home of the §5 toolkit and the precise dual of the twin non-covering question.
5. **Computation** — smallest admissible center past $L$, and gaps between consecutive admissible centers vs. the average $1/\text{density}$. Makes wall (A) tangible; links to maximal prime gaps and Cramér-type heuristics.

---

**Organizing takeaway.** The centers line (§§2–4), the omission/Fermat line (§5), the finite-set split (§7), and the coefficient/resultant line (§§8–9) are encodings of one move — manufacture an admissible / small-factor-free object and hope it is prime. All reduce to *admissible center in a short window*. Their modular content is exactly the singular series — the understood half (§10) — and the target is held shut by short-interval existence (A) and, beneath it, by the factor-count/parity barrier (B), which is a proven limitation of Type-I methods rather than a blanket impossibility.
