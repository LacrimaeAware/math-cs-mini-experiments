# First Composite Distance from a Primorial (bidirectional nearest-prime search)
# PyCharm-ready: edit CONFIG and run.

# =========================
# CONFIG — EDIT THESE
# =========================
LAST_PRIMES = []         # e.g. [7, 11, 13]; if empty, use all primes ≤ LAST_PRIME_MAX
LAST_PRIME_MAX = 43      # build primorial endpoints for all primes ≤ this
MAX_STEPS = 200000       # safety cap on how many neighbor primes we examine
VERBOSE = True
# =========================

from typing import List, Tuple, Optional

# Project-local shared sieve (see prime_lib.py at repo root). NOTE: this script
# deliberately keeps its own deterministic Miller-Rabin (fixed bases) below.
import sys
import pathlib
sys.path.append(next(str(p) for p in pathlib.Path(__file__).resolve().parents if (p / "prime_lib.py").exists()))
from prime_lib import sieve_primes

# ---------- small sieve ----------
# sieve_primes() is imported from prime_lib (see above).

# ---------- Miller–Rabin probable prime (good for large ints) ----------
_SMALL_PRIMES = [2,3,5,7,11,13,17,19,23,29,31,37]

def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    # write n-1 = 2^s * d
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # bases: deterministic for 64-bit; fine as a strong filter more generally
    for a in (2, 3, 5, 7, 11, 13, 17):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def is_prime_int(n: int) -> bool:
    return is_probable_prime(n)

# ---------- next/prev prime search around N ----------
def next_prime_after(n: int) -> int:
    c = n + 1
    if c % 2 == 0:
        c += 1
    while True:
        if is_probable_prime(c):
            return c
        c += 2

def prev_prime_before(n: int) -> Optional[int]:
    c = n - 1
    if c < 2:
        return None
    if c % 2 == 0:
        c -= 1
    while c >= 2:
        if is_probable_prime(c):
            return c
        c -= 2
    return None

# ---------- primorial ----------
def build_primorial_upto(last_prime: int) -> Tuple[int, List[int]]:
    S = sieve_primes(max(1000, last_prime * 50))
    plist = [p for p in S if p <= last_prime]
    if not plist or plist[-1] != last_prime:
        raise ValueError(f"last_prime={last_prime} not prime or out of sieve range.")
    N = 1
    for p in plist:
        N *= p
    return N, plist

# ---------- core: first composite distance ----------
def first_composite_distance_for_primorial(last_prime: int,
                                           max_steps: int = 200000,
                                           verbose: bool = False):
    N, plist = build_primorial_upto(last_prime)

    # initialize nearest primes on both sides
    p_up = next_prime_after(N)
    p_dn = prev_prime_before(N)

    # log visited (P, distance, kind) with kind in {"up","down"}
    visited = []

    steps = 0
    first_comp_distance: Optional[int] = None
    first_comp_prime: Optional[int] = None
    first_comp_side: Optional[str] = None

    while steps < max_steps:
        steps += 1

        # compute current distances (use "infinite" if missing one side)
        d_up = p_up - N if p_up is not None else 1 << 62
        d_dn = N - p_dn if p_dn is not None else 1 << 62

        # choose nearer side; tie-breaker: prefer down first (arbitrary) or choose up—either is fine
        if d_dn <= d_up:
            P = p_dn
            d = d_dn
            side = "down"
        else:
            P = p_up
            d = d_up
            side = "up"

        # record
        visited.append((P, d, side))

        # classify distance: composite if >=2 and not prime; (d=1 is neither prime nor composite)
        is_d_prime = (d >= 2 and is_prime_int(d))
        is_d_composite = (d >= 2 and not is_d_prime)

        if is_d_composite:
            first_comp_distance = d
            first_comp_prime = P
            first_comp_side = side
            break

        # advance the chosen side to the next prime farther away
        if side == "down":
            p_dn = prev_prime_before(p_dn) if p_dn is not None else None
        else:
            p_up = next_prime_after(p_up) if p_up is not None else None

        # if both sides exhausted (theoretically only for tiny N), stop
        if p_up is None and p_dn is None:
            break

    if verbose:
        print(f"[last_prime={last_prime}] primorial N={N}")
        print(f"  steps={steps}")
        if first_comp_distance is not None:
            print(f"  FIRST COMPOSITE DISTANCE: d={first_comp_distance} (prime={first_comp_prime}, side={first_comp_side})")
        else:
            print("  No composite distance found within MAX_STEPS.")
        # show a few early visits
        show = visited[:10]
        if show:
            print("  first 10 visits (P, |P-N|, side):")
            for row in show:
                print("   ", row)
        print()

    return {
        "last_prime": last_prime,
        "primorial_N": N,
        "steps": steps,
        "first_composite_distance": first_comp_distance,
        "first_composite_prime": first_comp_prime,
        "first_composite_side": first_comp_side,
        "visited_prefix": visited[:50],  # small preview
    }

# ---------- driver ----------
def main():
    if LAST_PRIMES:
        endpoints = LAST_PRIMES
    else:
        endpoints = sieve_primes(LAST_PRIME_MAX)

    if not endpoints:
        print("No endpoints to test.")
        return

    results = []
    for lp in endpoints:
        try:
            res = first_composite_distance_for_primorial(lp, max_steps=MAX_STEPS, verbose=VERBOSE)
            results.append(res)
        except Exception as e:
            print(f"[WARN] last_prime={lp}: {e}")

    print("=== Summary ===")
    print(f"Primorials tested: {len(results)}")
    have = [r for r in results if r['first_composite_distance'] is not None]
    if have:
        avg_d = sum(r['first_composite_distance'] for r in have) / len(have)
        print(f"Average first composite distance (over found): {avg_d:.2f}")
    else:
        print("No composite distance found within MAX_STEPS for any case.")

if __name__ == "__main__":
    main()
