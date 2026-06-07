# The modular ceiling

Work the [coefficient form](06-coefficient-form-and-resultant-detector.md) mod a
small prime $\ell$. With $F(x)=\prod_i(x+a_i)=\sum_k\sigma_k x^{n-k}$,
$$F(x)\equiv\prod_{r\bmod\ell}(x+r)^{m_r}\pmod{\ell},\qquad m_r=\#\{i:a_i\equiv r\}.$$
So every coefficient mod $\ell$ is just bookkeeping of the residue counts $\{m_r\}$
— the distribution of the numbers across classes mod $\ell$ is the *entire* modular
content. This cannot see twinness: a real twin differs by *exactly $2$ as
integers*, but mod $\ell$ that becomes "two residues differing by $2$ mod $\ell$,"
a weaker thing. Mod $5$, the twins $11,13$ sit at $1,3$ and the non-twins $7,19$
sit at $2,4$ — both "differ by $2$ mod $5$." Once a set covers more than half the
residues mod $\ell$ (any large set does), some pair differs by $2$ mod $\ell$, so
$\operatorname{Res}\equiv0\pmod\ell$ for essentially every large set, twin-free or
not. The twin/twin-free signal survives only in the *exact integer value*, never in
any reduction mod $\ell$.

So the modular route bottoms out not by being "vapid by definition," but because
reduction mod small primes is precisely the operation that discards gap *size*:
differing-by-$2$ is an archimedean fact, and mod $\ell$ keeps only
differing-by-$2$-mod-$\ell$. Assembled over all small $\ell$, the residue-count
data is exactly the **singular series** — the twin-prime constant
$C_2=\prod_{p>2}\!\big(1-\tfrac1{(p-1)^2}\big)$ — which converges, is computable,
and is the local density of twin-admissible residues. Hardy–Littlewood: the count
is *that modular constant* times an archimedean term $\sim N/(\log N)^2$. Pushing
coefficient/modular algebra harder walks straight into $C_2$ (the solved half); the
difficulty sits, provably, in the complementary non-modular term.
