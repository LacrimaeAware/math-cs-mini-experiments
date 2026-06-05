#!/usr/bin/env python3
"""
Wheel border-merge statistics.

Given a base prime set S (e.g., [2,3,5,7]), we consider the wheel modulo P=prod(S).
Then we extend to modulus P*q where q is a new prime (default: next prime after max(S)),
*without* sieving by q, and finally sieve by q.
Each removed multiple of q merges its left and right bordering gaps a,b into a+b.
We collect frequencies of unordered pairs {a,b} over all removals.

This reproduces stats like:
"gap 2 merges with gap 4" -> count / (#removals)

Run examples:
    python wheel_merges.py
    python wheel_merges.py --primes 2 3 5 7 --next 11
    python wheel_merges.py --primes 2 3 5 7 11 --next 13
"""

from __future__ import annotations
import math
import argparse
from collections import Counter
from typing import List, Tuple, Dict

# Project-local shared helpers (see prime_lib.py at repo root).
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from prime_lib import is_prime, next_prime_after, primorial


# is_prime(), next_prime_after() and primorial() are imported from prime_lib.


def totatives_mod(P: int, primes: List[int]) -> List[int]:
    """Sorted residues in [0,P) coprime to all primes in the list."""
    res = []
    for a in range(P):
        ok = True
        for p in primes:
            if a % p == 0:
                ok = False
                break
        if ok:
            res.append(a)
    return res


def gap_word_from_totatives(tots: List[int], P: int) -> List[int]:
    """Cyclic gaps between consecutive totatives."""
    if not tots:
        return []
    gaps = []
    m = len(tots)
    for i in range(m):
        a = tots[i]
        b = tots[(i + 1) % m]
        gap = (b - a) % P
        gaps.append(gap)
    return gaps


def border_merge_stats(base_primes: List[int], q: int | None = None
                      ) -> Tuple[int, Counter, Counter]:
    """
    Returns:
        q_used,
        unordered_pair_counts: Counter({(min_gap,max_gap): count})
        merged_gap_counts: Counter({a+b: count})
    """
    base_primes = sorted(base_primes)
    P = primorial(base_primes)
    q_used = q if q is not None else next_prime_after(base_primes[-1])

    # Totatives mod P and their gaps (base wheel)
    tots_P = totatives_mod(P, base_primes)
    gaps_P = gap_word_from_totatives(tots_P, P)
    m = len(tots_P)  # number of candidates per base period

    # Lift totatives to modulus P*q without sieving by q:
    # candidates are {a + tP} for each a in tots_P, t=0..q-1
    PQ = P * q_used

    # For each base totative a, exactly one lift is divisible by q_used:
    # find t such that a + tP ≡ 0 (mod q)
    invP = pow(P, -1, q_used)  # modular inverse (Python 3.8+)
    kill_positions = []  # positions among the lifted candidates (in order)

    # We need the lifted candidates in sorted cyclic order to locate bordering gaps.
    # Generate all candidates, mark which are killed, then sort.
    candidates = []
    killed_set = set()

    for a in tots_P:
        t_kill = (-a * invP) % q_used
        for t in range(q_used):
            x = a + t * P
            candidates.append(x)
            if t == t_kill:
                killed_set.add(x)

    candidates.sort()
    # Sanity: number of kills should be m
    assert len(killed_set) == m

    # Now compute gaps on the unsieved lifted wheel (still only sieved by base primes)
    # and record bordering gaps around each killed candidate.
    pair_counts = Counter()
    merged_counts = Counter()

    L = len(candidates)
    # Map candidate value -> index in sorted list
    idx = {x: i for i, x in enumerate(candidates)}

    for x in killed_set:
        i = idx[x]
        left_i = (i - 1) % L
        right_i = (i + 1) % L
        left_gap = (candidates[i] - candidates[left_i]) % PQ
        right_gap = (candidates[right_i] - candidates[i]) % PQ

        a, b = sorted((left_gap, right_gap))
        pair_counts[(a, b)] += 1
        merged_counts[a + b] += 1

    return q_used, pair_counts, merged_counts


def format_pair_stats(pair_counts: Counter, total: int) -> str:
    lines = []
    for (a, b), c in sorted(pair_counts.items()):
        pct = 100.0 * c / total
        if a == b:
            lines.append(f"“gap {a} merges with gap {b}” → {c}/{total} = {pct:.2f}%")
        else:
            lines.append(f"“gap {a} merges with gap {b}” → {c}/{total} = {pct:.2f}%")
    return "\n\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primes", nargs="+", type=int,
                    default=[2, 3, 5, 7],
                    help="Base primes for the wheel (space-separated).")
    ap.add_argument("--next", type=int, default=None,
                    help="Next prime q to sieve by. Default: next prime after max(primes).")
    args = ap.parse_args()

    q_used, pair_counts, merged_counts = border_merge_stats(args.primes, args.next)
    total = sum(pair_counts.values())

    print(f"Base primes: {sorted(args.primes)}")
    print(f"Base modulus P: {primorial(sorted(args.primes))}")
    print(f"Next prime sieved q: {q_used}")
    print(f"#removals: {total}")
    print()
    print(format_pair_stats(pair_counts, total))


if __name__ == "__main__":
    main()
