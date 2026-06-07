"""
nonlinear_arithmetic_sets.py — signed-sum sets that cover the coprimes mod 210.

Searches for small structured sets A whose signed subset sums (each element +/-)
are all coprime to 210, and whose positive sums hit every coprime residue in
[1, 209]. Uses a fixed scaffold A = 7*{3,5,6,15} union {30*t1..30*t4} and sweeps
the t_i. Reports the best coverage (fewest missing coprimes) found.
"""

import itertools, math, sys, time

M = 210

def is_coprime(a, m=M):
    return math.gcd(abs(a), m) == 1

def signed_sums(A):
    out = {0}
    for a in A:
        out = {s + a for s in out} | {s - a for s in out}
    return out

def targetB_210(A):
    """
    Target B for 210:
      - every signed sum coprime to 210
      - positive signed sums contain all coprimes 1..209
    Returns (ok, missing, extras).
    """
    sums = signed_sums(A)

    # all sums must be coprime to 210
    for s in sums:
        if not is_coprime(s):
            return False, None, None

    pos = {s for s in sums if s > 0}
    small = {n for n in range(1, M) if is_coprime(n)}

    missing = sorted(small - pos)
    extras  = sorted(pos - small)
    ok = (len(missing) == 0)
    return ok, missing, extras

def search_structured_210(T=25):
    """
    Exhaustively search 8-term sets of the form:
      A = 7*{3,5,6,15}  union  {30*t1,30*t2,30*t3,30*t4}
    with t_i in [1..T], t_i not divisible by 7.
    """
    B = [3, 5, 6, 15]
    base = [7*b for b in B]  # [21,35,42,105]

    candidates_t = [t for t in range(1, T+1) if t % 7 != 0]
    combos = list(itertools.combinations_with_replacement(candidates_t, 4))
    total = len(combos)

    print(f"T={T}, candidate t's={candidates_t}")
    print(f"Total Y-multisets to test: {total}")

    best_missing = len([n for n in range(1, M) if is_coprime(n)])
    best = None

    start = time.time()

    for idx, Y in enumerate(combos, 1):
        A = base + [30*t for t in Y]
        ok, missing, extras = targetB_210(A)

        if missing is not None:
            mcount = len(missing)
            if mcount < best_missing:
                best_missing = mcount
                best = (Y, missing, extras)
                print(f"\nNew best at {idx}/{total}: missing={best_missing}")
                print(f"  Y = {Y}")
                print(f"  A = {A}")
                if ok:
                    print("\nPERFECT COVER FOUND.")
                    return A, missing, extras

        # simple progress bar
        if idx % 500 == 0 or idx == total:
            frac = idx / total
            bar_len = 30
            filled = int(bar_len * frac)
            bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"
            elapsed = time.time() - start
            sys.stdout.write(f"\r{bar} {idx}/{total} elapsed={elapsed:.1f}s  best_missing={best_missing}")
            sys.stdout.flush()

    print("\n\nNo perfect cover for this T.")
    if best is not None:
        Y, missing, extras = best
        print(f"Best Y = {Y}")
        print(f"Missing count = {len(missing)}")
        print(f"Missing (first 20) = {missing[:20]}")
    return None

# Example run:
search_structured_210(T=55)
