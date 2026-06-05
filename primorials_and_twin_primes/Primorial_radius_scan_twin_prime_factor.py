"""
Primorial_radius_scan_twin_prime_factor.py — twin-prime factors within a radius.

For each n in [MIN_N, MAX_N], scan offsets d = 1..RADIUS on both sides of the
n-th primorial P (N = P + d and/or P - d), factor each N, and record whether a
twin-prime factor appears (excluding factors shared with gcd(P, d)). Produces a
leaderboard of which offsets d most often yield a twin-prime factor.
"""

# Project-local shared helpers + output directory (see prime_lib.py at repo root).
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from prime_lib import ensure_output_dir, sieve_primes

import os, json, time, math
from collections import defaultdict
from math import gcd, isqrt

from sympy import primorial, prime, isprime, factorint

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =========================
# YOU EDIT THIS SECTION
# =========================
MIN_N = 1
MAX_N = 20
RADIUS = 500

DO_PLUS = True
DO_MINUS = True

SMALL_N_CUTOFF = 10**12
TRIAL_PRIME_LIMIT = 200000

PRINT_EVERY_SECONDS = 10
TOP_K = 10               # leaderboard size
TOP_K_LISTS = 10         # how many top offsets also show success/fail arrays

CACHE_DIR = str(ensure_output_dir("primorial_squares_cache"))
# =========================


def ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def primorial_index(n: int) -> int:
    return int(primorial(n, nth=True))


def nth_prime(n: int) -> int:
    return int(prime(n))


def sanity_check_index_primorial():
    known = {1: 2, 2: 6, 3: 30, 4: 210, 5: 2310}
    for n, v in known.items():
        got = primorial_index(n)
        if got != v:
            raise RuntimeError(
                f"[SANITY FAIL] primorial_index({n})={got}, expected {v}"
            )


# sieve_primes() is imported from prime_lib (see top of file).


PRIMES_TRIAL = sieve_primes(TRIAL_PRIME_LIMIT)


def trial_factor_small(n):
    facs = {}
    x = n
    for p in PRIMES_TRIAL:
        if p * p > x:
            break
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            facs[p] = e
    if x > 1:
        facs[x] = facs.get(x, 0) + 1
    return facs


def primes_from_gcd(g):
    if g <= 1:
        return set()
    facs = trial_factor_small(g)
    return set(facs.keys())


def has_twin_prime_factor(factors, exclude_set):
    for r in factors.keys():
        if r in exclude_set:
            continue
        if isprime(r - 2) or isprime(r + 2):
            return True
    return False


def factor_N(N):
    aN = abs(N)
    if aN < SMALL_N_CUTOFF:
        return trial_factor_small(aN), True, False
    facs = factorint(aN)
    return facs, True, False


def top_offsets(hits, total, unknown, radius, top_k):
    rows = []
    for d in range(1, radius + 1):
        known = total[d] - unknown[d]
        pct = hits[d] / known if known > 0 else 0.0
        rows.append((d, pct, hits[d], known))
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return rows[:top_k]


def run():
    ensure_dir()
    sanity_check_index_primorial()

    plus_hits = defaultdict(int)
    plus_total = defaultdict(int)
    plus_unknown = defaultdict(int)

    minus_hits = defaultdict(int)
    minus_total = defaultdict(int)
    minus_unknown = defaultdict(int)

    # NEW: success/fail lists
    plus_success_ns = defaultdict(list)
    plus_fail_ns = defaultdict(list)

    minus_success_ns = defaultdict(list)
    minus_fail_ns = defaultdict(list)

    tasks = []
    for n in range(MIN_N, MAX_N + 1):
        if DO_PLUS:
            for d in range(1, RADIUS + 1):
                tasks.append((n, +1, d))
        if DO_MINUS:
            for d in range(1, RADIUS + 1):
                tasks.append((n, -1, d))

    iterator = tasks
    if tqdm is not None:
        iterator = tqdm(tasks, desc="Scanning primorial radius", unit="case")

    last_print = time.time()
    t0 = time.time()

    for idx, (n, sign, d) in enumerate(iterator, start=1):
        P = primorial_index(n)
        N = P + sign * d

        g = gcd(P, d)
        exclude_set = primes_from_gcd(g)

        facs, fully, timed_out = factor_N(N)
        found_twin = (not timed_out) and has_twin_prime_factor(facs, exclude_set)

        if sign == +1:
            plus_total[d] += 1
            if timed_out:
                plus_unknown[d] += 1
            else:
                if found_twin:
                    plus_hits[d] += 1
                    plus_success_ns[d].append(n)
                else:
                    plus_fail_ns[d].append(n)

        else:
            minus_total[d] += 1
            if timed_out:
                minus_unknown[d] += 1
            else:
                if found_twin:
                    minus_hits[d] += 1
                    minus_success_ns[d].append(n)
                else:
                    minus_fail_ns[d].append(n)

        now = time.time()
        if now - last_print >= PRINT_EVERY_SECONDS:
            elapsed = now - t0
            rate = idx / elapsed if elapsed else 0.0
            print(f"\n[hb {idx}/{len(tasks)}] rate={rate:.2f} cases/s")
            last_print = now

    # Final leaderboards with lists
    if DO_PLUS:
        plus_top = top_offsets(plus_hits, plus_total, plus_unknown, RADIUS, TOP_K)
        print("\n=== FINAL PLUS leaders ===")
        for i, (d, pct, hits, known) in enumerate(plus_top, start=1):
            print(f"d={d:3d} pct={pct:6.2%} hits={hits:3d} known={known:3d}")
            if i <= TOP_K_LISTS:
                print(f"   success n's: {plus_success_ns[d]}")
                print(f"   fail n's:    {plus_fail_ns[d]}")

    if DO_MINUS:
        minus_top = top_offsets(minus_hits, minus_total, minus_unknown, RADIUS, TOP_K)
        print("\n=== FINAL MINUS leaders ===")
        for i, (d, pct, hits, known) in enumerate(minus_top, start=1):
            print(f"d={d:3d} pct={pct:6.2%} hits={hits:3d} known={known:3d}")
            if i <= TOP_K_LISTS:
                print(f"   success n's: {minus_success_ns[d]}")
                print(f"   fail n's:    {minus_fail_ns[d]}")


if __name__ == "__main__":
    run()
