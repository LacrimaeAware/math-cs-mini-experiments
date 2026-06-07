"""
clustering_prime_products.py — partition the first k primes to balance products.

Exact branch-and-bound optimizer. Use each of the first k primes exactly once and
split them into groups subject to:
  * each group's product <= (next prime after p_k)^2,
  * the only allowed singleton group is the largest prime p_k,
then minimize the average pairwise absolute difference between group products
(i.e. make the group products as close to one another as possible).
"""

from math import prod
import sys
sys.setrecursionlimit(10000)

# ---------- minimal prime utilities (no external deps) ----------
def first_k_primes_and_next(k: int):
    if k <= 0:
        return [], None
    import math
    # over-approx upper bound for k-th prime
    n = max(20, int(k * (math.log(k) + math.log(max(2, math.log(k)))) + 50))
    while True:
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for p in range(2, int(n ** 0.5) + 1):
            if sieve[p]:
                start = p * p
                sieve[start:n + 1:p] = [False] * (((n - start) // p) + 1)
        primes = [i for i, b in enumerate(sieve) if b]
        if len(primes) >= k + 1:
            return primes[:k], primes[k]
        n *= 2  # grow and try again

# ---------- objective helpers ----------
def sum_pairwise_abs(sorted_products):
    # For sorted p1<=...<=pg:
    # sum_{i<j} (pj - pi) = sum_i p_i * (2*i - g - 1)
    g = len(sorted_products)
    if g <= 1:
        return 0
    s = 0
    for idx, p in enumerate(sorted_products, start=1):
        s += p * (2 * idx - g - 1)
    return s

def avg_pairwise(products):
    g = len(products)
    if g <= 1:
        return 0.0
    prods_sorted = sorted(products)
    total = sum_pairwise_abs(prods_sorted)
    return total * 2 / (g * (g - 1))

# ---------- exact search (no over-pruning) ----------
def optimal_partition_for_k(k: int):
    primes, next_p = first_k_primes_and_next(k)
    cap = next_p * next_p
    p_max = primes[-1] if primes else None
    desc = list(reversed(primes))  # place larger primes first for stronger pruning

    best_avg = float("inf")
    best_groups = None

    # groups is a list of lists of primes
    # We'll explore all ways to place desc[idx] either into an existing group (if product <= cap)
    # or into a new group; we only enforce "only p_max may be singleton" at leaves.
    from functools import lru_cache

    def state_key(groups, idx):
        # Canonical state for memo: idx plus multiset of (product, size, contains_pmax)
        triples = []
        for g in groups:
            triples.append((prod(g), len(g), 1 if (p_max in g) else 0))
        triples.sort()
        return (idx, tuple(triples))

    @lru_cache(maxsize=None)
    def memo_value(key):
        # store best achievable avg from this state (for pruning identical states)
        return None  # placeholder; real values are stored in outer dict 'visited_best'

    visited_best = {}  # key -> best achievable avg from that state (for pruning)

    def search(idx, groups):
        nonlocal best_avg, best_groups

        # Prune identical states we have seen with a <= known bound
        key = state_key(groups, idx)
        if key in visited_best:
            # If we already proved this state can't beat current best, skip
            if visited_best[key] >= best_avg:
                return
        # Record the *current* best bound at this state
        visited_best[key] = best_avg

        if idx == len(desc):
            # Enforce singleton rule at leaf: only p_max may be singleton
            for g in groups:
                if len(g) == 1 and g[0] != p_max:
                    return
            products = [prod(g) for g in groups]
            cur = avg_pairwise(products)
            if cur < best_avg:
                best_avg = cur
                # store a deep copy (sorted by product for readability)
                best_groups = [sorted(g) for g in groups]
            return

        p = desc[idx]

        # Try to place into existing groups (prefer smaller products first)
        # Also avoid duplicate branches where two groups have the same product & size.
        seen_targets = set()
        groups_info = [(i, prod(g), len(g)) for i, g in enumerate(groups)]
        for i, gp, glen in sorted(groups_info, key=lambda t: (t[1], t[2])):
            newp = gp * p
            sig = (gp, glen)  # symmetry breaker
            if sig in seen_targets:
                continue
            seen_targets.add(sig)
            if newp <= cap:
                groups[i].append(p)
                search(idx + 1, groups)
                groups[i].pop()

        # Start a new group with p (always allowed; final rule checked at leaf)
        groups.append([p])
        search(idx + 1, groups)
        groups.pop()

    search(0, [])

    return primes, next_p, cap, best_avg if best_groups is not None else None, best_groups

def pretty_print(k, primes, next_p, cap, opt, groups):
    print(f"k = {k}")
    print("primes:", primes)
    print(f"next prime: {next_p}  cap = {cap}")
    if groups is None:
        print("No valid partition found under the constraints.")
        return
    prods = [prod(g) for g in groups]
    print(f"Optimal average pairwise distance: {opt}")
    print(f"Number of groups: {len(groups)}")
    print("Group products (sorted):", sorted(prods))
    print("Groups:")
    for g in sorted(groups, key=lambda G: prod(G)):
        print(" ", g, "→", prod(g))

if __name__ == "__main__":
    # k=12 and k=14 are exact & fast; k=16 may be slow; k=20 is generally too large for exact.
    for k in (8, 1):
        primes, next_p, cap, opt, groups = optimal_partition_for_k(k)
        pretty_print(k, primes, next_p, cap, opt, groups)
