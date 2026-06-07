# Coefficient form and the resultant detector

## The product / coefficient form

Write each prime via base-plus-gap and take the product. For $\{2,23,29\}$ (gaps
$0,21,27$ from base $2$):
$$p\,(p+21)(p+27)=p^3+48\,p^2+567\,p,\qquad 48=21+27,\;\;567=21\cdot27,$$
checking at $p=2$: $8+192+1134=1334=2\cdot23\cdot29$. The coefficients are the
**elementary symmetric functions of the gaps**: each is a power of $p$ times a
product of some subset of the gaps. The base prime is "excluded", its gap is $0$,
so it only ever contributes a power of $p$ and never enters a cross-term (which is
also the "$p$ appears twice" feeling: $p$ is both the pulled-out factor and the
additive piece inside each binomial). Note two distinct objects: gaps-as-offsets
gives symmetric functions of the *distances*; adding the base to each prime,
$\prod_i(2+q_i)$, gives $2^k$ times subset-products of the *primes*.

## Detecting twins algebraically: shift-and-compare

Let $f(x)=\prod_i(x-q_i)$ (primes as roots; same data as the product above).

> twin-free $\iff$ none of $q_i+2$ is in the set $\iff f(q_i+2)\ne0$ for all $i$.

Picture: lay the set down and a copy shifted right by $2$ on top; a twin pair is an
overlap point. Examples: $\{5,7\}$ gives $f(7)=0$ (caught); $\{5,11\}$ gives
$f(7)=-8,\,f(13)=16$ (no overlap). The number packaging this is the **resultant**
$\operatorname{Res}\!\big(f(x),f(x+2)\big)=\prod_i f(q_i+2)$, zero iff a twin pair
exists, and a polynomial in the coefficients of $f$.

Caveat on what this *is*: written as $\prod_i f(q_i+2)$ it is the inert
"factor-testing" form, an exhaustive pairwise scan wearing a product sign,
equivalent to the very condition it tests, hence a reformulation rather than a
lever. But the *instinct* is correct and is exactly how the problem is posed
analytically: the twin count is the autocorrelation $\sum_n\Lambda(n)\Lambda(n+2)$,
the prime indicator against its own shift by $2$. The resultant is the finite,
exact shadow of that sum. Reducing it mod small primes runs into
[the modular ceiling](07-the-modular-ceiling.md).
