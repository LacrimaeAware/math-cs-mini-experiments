# Euclid, and the finite-twins split

## Why Euclid's modular argument succeeds where this one stalls

Euclid uses modular reasoning and works — the point is *what he asks of it*. He
forms $N=p_1\cdots p_n+1$, concludes $N$ is coprime to every $p_i$, and stops. He
never claims $N$ is prime, and it often isn't:
$2\cdot3\cdot5\cdot7\cdot11\cdot13+1=30031=59\times509$. All he needs is "$N$ has
*some* prime factor, which must be new," and every integer $>1$ has one. His
conclusion does not care whether $N$ has one prime factor or twelve.

Twin primes need $m-1$ and $m+1$ each to have *exactly one* prime factor.
Coprimality buys "$m\pm1$ have only large factors," but a number with only large
factors can still be a product of two large primes. So what coprimality hands
Euclid for free — "divisible by something new," true regardless of factor count —
is not what twins need — "exactly one prime factor," a statement about the *number*
of factors, which congruence data does not encode. That is the precise content
behind "coprime $\ne$ prime," and it is about the *kind of conclusion*
(factor-count-blind vs factor-count-sensitive), not about bounds. ("Parity" is
shorthand for that factor count mod $2$; the claim is that residue information is
blind to it — see [the two walls and parity](08-the-two-walls-and-parity.md).)

## The finite-twins assumption and the A/B split

Assume finitely many twins; split the primes $P=A\sqcup B$ with $B$ the twins, $A$
the rest, and form e.g. $N=\prod_A+\prod_B$. Mod any $q\in P$: if $q\in A$ then
$N\equiv\prod_B\not\equiv0$, symmetrically for $q\in B$, so $N$ is coprime to all
of $P$ — it has a prime factor outside $P$. This is Euclid with labels, and it
leaks: reducing $N$ mod $q$ keeps the residue and discards the label — it has no
idea whether $q$ was a twin. The labeling can steer which residue class $N$ lands
in, but no choice reaches into the primality of $N$'s *neighbors*, and a twin pair
is a fact about neighbors. The output is a new prime, not a new twin.

Under the "$P$ = all primes up to the last twin $T_{\max}$" version, that is not
even a contradiction: a prime beyond $T_{\max}$ is guaranteed (primes are infinite
whether or not twins are), and the assumption restricts twins, not primes. The
whole remaining distance is "new prime ⟹ new twin," exactly the step finiteness
gives no grip on; so the bounded-range version affords the same single fact as the
arbitrary-range one. The sharpest honest form of "twins are finite" is: *every
admissible center $m>T_{\max}$ has a composite (large-factored) neighbor* — every
abundant twin-coprime center is spoiled. But spoiling is a primality fact, not
chosen by residues, so no modular contradiction follows however tightly the range
is bounded.

The trivial deduction "no twins ⟹ consecutive gaps $\ge4$" is true and nearly
empty: the average gap near $x$ is $\sim\log x$, which dwarfs $4$ long before
anything interesting.
