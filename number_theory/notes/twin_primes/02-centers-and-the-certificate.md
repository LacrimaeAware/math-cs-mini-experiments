# Centers and the certificate

(Notation — $W_L$, $L$-admissible, $\nu(L)$ — is in the [index](README.md).)

## The certificate

> Let $L\ge2$ and $m$ be $L$-admissible with $L<m-1$ and $m+1\le L^2$. Then $m-1$
> and $m+1$ are both prime.

*Proof.* Admissibility ⟹ $m\pm1$ have no prime factor $\le L$. A composite $n$ has
a prime factor $\le\sqrt n$; $m\pm1\le L^2$ gives $\sqrt{m\pm1}<L$, so neither is
composite. $\square$

So admissible centers in the window $(L,\,L^2)$ are exactly twin-prime centers
(the lower bound $m-1>L$ omits small pairs where a member equals a sieve prime).
This is the same engine as the **$X+P$ construction** under the dictionary
$X+P-2,\,X+P \leftrightarrow$ center $m=X+P-1$: forbidding $P\equiv-X,-X+2\pmod Q$
for all $Q<X$ is exactly $m\not\equiv\pm1\pmod Q$. The center coordinate makes the
symmetry $m\mapsto(m-1,m+1)$ and the link to the singular series explicit.

## The difference rule

> Let $X,Y$ be $L$-admissible, $D=Y-X$, and $q\le L$ prime. If $q\mid X$ or
> $q\mid Y$, then $D\not\equiv\pm1\pmod q$. If $q\nmid X$ and $q\nmid Y$, then $q$
> obstructs $D$ exactly when $Y\equiv X\pm1\pmod q$.

*Proof.* If $q\mid X$ then $D\equiv Y\not\equiv\pm1$; $\{\pm1\}$ is closed under
negation, so $q\mid Y$ is identical. Otherwise $D\equiv\pm1$ is the literal
obstruction. $\square$

Two corrections fall out as mathematics: "divides neither center ⟹ $q\nmid D$" is
false ($X\equiv Y$ gives $q\mid D$), but harmless, since the obstructed residues
are $\pm1$ and $D\equiv0$ is not among them. The same fact makes the reference
choice $A\equiv0\pmod{W_L}$ clean: $m\mapsto m-A\equiv m\pmod{W_L}$ preserves
admissibility.

Trying to *build* such centers by congruences runs into
[the lattice obstruction](03-the-lattice-obstruction.md).
