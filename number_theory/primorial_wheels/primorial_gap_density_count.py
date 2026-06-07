#!/usr/bin/env python3
"""
Search worst deviations of a model period for p-classes in a primorial modulo.

Definitions:
- Base prime set: primes[0..N-1], product M = p_1 * ... * p_N.
- "p-class" for primes[i]:
    numbers n mod M such that:
        n % p_i == 0
        and n % q != 0 for all primes[j] with j < i.

We test a model "period" D_p for each p in a chosen range and look for
intervals [start, start+L] (mod M) where

    deviation = actual_count - ceil((L+1)/D_p)

is maximized. That tells you how far the actual count can fall under the ceiling model.
"""

from math import ceil
import sys
import time

# ---------- basic prime utilities ----------

def first_primes(k):
    """Return the first k primes (simple trial division)."""
    primes = []
    n = 2
    while len(primes) < k:
        is_p = True
        for q in primes:
            if q * q > n:
                break
            if n % q == 0:
                is_p = False
                break
        if is_p:
            primes.append(n)
        n += 1
    return primes


# ---------- p-class positions in the modulo ----------

def p_class_positions(primes, idx):
    """
    primes: list of primes [p1, p2, ..., pK]
    idx: index of the prime we care about (0-based)
    Returns (positions, M):
      - positions: sorted list of x in [0, M) such that
          x % p == 0 and x not divisible by any smaller prime in primes[:idx]
      - M: modulus = product of primes[:idx+1]
    """
    p = primes[idx]
    smaller = primes[:idx]
    M = 1
    for q in primes[:idx+1]:
        M *= q

    pos = []
    # only need to check multiples of p
    for x in range(0, M, p):
        if all(x % q != 0 for q in smaller):
            pos.append(x)
    return pos, M


# ---------- worst deviation search for one prime ----------

def worst_deviation_for_prime(primes, idx, period, L_max_factor=1.0):
    """
    For prime primes[idx] with model period 'period',
    search intervals of length L up to L_max_factor * M (default = M),
    and find the maximal deviation:
        deviation = actual_count - ceil((L+1)/period)

    Returns (info, M) where info is a dict with keys:
      'dev'   : maximal deviation
      'L'     : interval length achieving it
      'start' : start position modulo M
      'count' : actual count of class-p hits in that interval
    """
    p = primes[idx]
    pos, M = p_class_positions(primes, idx)
    L_max = int(L_max_factor * M)

    k = len(pos)
    if k == 0:
        return {
            'p': p,
            'dev': None,
            'L': None,
            'start': None,
            'count': 0,
            'period': period,
            'M': M,
            'num_hits': 0
        }, M

    # crude guard: full O(k^2) scan will be insane if k is huge
    if k > 2000:
        print(f"  [WARN] p={p}, class size k={k} is large; "
              f"O(k^2) scan may be very slow.", file=sys.stderr)

    # allow wrap-around by duplicating positions shifted by M
    ext = pos + [x + M for x in pos]

    best = {'dev': None, 'L': None, 'start': None, 'count': None}

    # For each starting index i, expand j until length > L_max
    # This is still O(k^2) worst-case, but fine for small k.
    last_progress = -1
    for i in range(k):
        # progress for outer loop (per prime)
        progress = int((i / max(1, k-1)) * 100)
        if progress != last_progress and progress % 5 == 0:
            print(f"    progress p={p}: {progress}%", end="\r", flush=True)
            last_progress = progress

        for j in range(i + 1, i + k + 1):
            L = ext[j - 1] - ext[i]
            if L > L_max:
                break
            count = j - i
            expected = ceil((L + 1) / period)
            dev = count - expected
            if best['dev'] is None or dev > best['dev']:
                best = {
                    'dev': dev,
                    'L': L,
                    'start': ext[i] % M,
                    'count': count
                }

    # clean up progress line
    print(" " * 40, end="\r", flush=True)

    best_info = {
        'p': p,
        'period': period,
        'dev': best['dev'],
        'L': best['L'],
        'start': best['start'],
        'count': best['count'],
        'M': M,
        'num_hits': k
    }
    return best_info, M


# ---------- driver over a range of primes ----------

def analyze_prime_set(P_MAX, p_min, p_max, period_mode="raw", L_max_factor=1.0):
    """
    P_MAX   : largest prime in the primorial we generate (2*3*...*P_MAX)
    p_min   : smallest prime whose class we analyze
    p_max   : largest prime whose class we analyze
              (inclusive, and must be <= P_MAX)
    period_mode:
        "prev" -> D_p = prev_prime * p, D_2 = 2
        "2p"   -> D_p = 2 * p for all p
        "raw"  -> D_p = 1 / D_raw(p) where
                  D_raw(p) = (1/p) * Π_{q<p} (1 - 1/q)
    L_max_factor: search interval lengths up to L_max_factor * M for each p

    Returns: (primes, results) where:
      - primes: base prime list up to P_MAX
      - results: list of info dicts (see worst_deviation_for_prime)
    """
    # get first primes until we hit P_MAX
    primes = []
    n = 2
    while True:
        # growing primes list until last == P_MAX
        is_p = True
        for q in primes:
            if q * q > n:
                break
            if n % q == 0:
                is_p = False
                break
        if is_p:
            primes.append(n)
            if n == P_MAX:
                break
        n += 1

    # map each prime to model period
    periods = {}

    if period_mode == "prev":
        for i, p in enumerate(primes):
            if i == 0:
                periods[p] = 2  # D_2 = 2
            else:
                periods[p] = primes[i-1] * p

    elif period_mode == "2p":
        for p in primes:
            periods[p] = 2 * p

    elif period_mode == "raw":
        # D_raw(p) = (1/p) * Π_{q<p} (1 - 1/q)
        # period = 1 / D_raw(p) = p / Π_{q<p} (1 - 1/q)
        prod_factor = 1.0  # Π_{q<p} (1 - 1/q), initially empty product
        for i, p in enumerate(primes):
            period = p / prod_factor
            periods[p] = period
            # update for next prime
            prod_factor *= (1.0 - 1.0 / p)
    else:
        raise ValueError("period_mode must be 'prev', '2p', or 'raw'")

    # select primes in [p_min, p_max]
    analyze_primes = [p for p in primes if p_min <= p <= p_max]

    results = []
    total = len(analyze_primes)
    print(f"Base primes up to {P_MAX}: {primes}")
    print(f"Analyzing p-classes for primes in [{p_min}, {p_max}] "
          f"({total} primes total)")
    print(f"Period mode: {period_mode}, L_max_factor: {L_max_factor}")
    print()

    for idx, p in enumerate(primes):
        if p not in analyze_primes:
            continue
        print(f"[{len(results)+1}/{total}] p={p}, period D_p={periods[p]}")
        t0 = time.time()
        info, M = worst_deviation_for_prime(primes, idx, periods[p], L_max_factor)
        t1 = time.time()
        print(f"  modulus M={M}, class size={info['num_hits']}")
        print(f"  worst deviation = {info['dev']}")
        print(f"  interval length L = {info['L']}")
        print(f"  start (mod M) = {info['start']}")
        print(f"  count in interval = {info['count']}")
        print(f"  time: {t1 - t0:.3f} s")
        print()
        results.append(info)

    return primes, results


# ---------- "MENU" / CONFIG SECTION ----------

if __name__ == "__main__":
    # === EDIT THESE ===

    # largest prime in the primorial (2*3*...*P_MAX)
    P_MAX = 23       # e.g. 17, 23, 29, etc. keep modest or runtime explodes

    # range of primes whose densities you want to inspect
    P_MIN_ANALYZE = 2  # smallest prime to analyze
    P_MAX_ANALYZE = 23  # largest prime to analyze (inclusive)

    # period model:
    #   "prev" -> D_p = prev_prime * p (with D_2 = 2)
    #   "2p"   -> D_p = 2 * p
    #   "raw"  -> D_p = 1 / D_raw(p) (correct asymptotic period)
    PERIOD_MODE = "raw"

    # how far to search relative to the modulus M for each p:
    #   L_max = L_max_factor * M
    L_MAX_FACTOR = 1.0  # 1.0 means up to length M

    # === RUN ===

    primes, results = analyze_prime_set(
        P_MAX=P_MAX,
        p_min=P_MIN_ANALYZE,
        p_max=P_MAX_ANALYZE,
        period_mode=PERIOD_MODE,
        L_max_factor=L_MAX_FACTOR,
    )
