#!/usr/bin/env python3
"""
orbit_primorials.py — reconstruct the coprimes of a primorial via "orbits".

For N = primorial(primes) and each squarefree divisor P of N (1 < P < N), form
centers C = A*(N/P) with gcd(A, P) = 1 and values |C +/- P^x| in [1, N]. The
union of all such values is compared against the true reduced residue system
(the phi(N) integers coprime to N) to measure how completely the orbit
construction covers it. Prints a per-P breakdown plus a global coverage check.
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from prime_lib import primes_up_to, primorial

import math
import itertools
from typing import List, Dict, Any, Set


# -------------------------------------------------------
# Basic helpers
# -------------------------------------------------------

# primes_up_to() and primorial() are imported from prime_lib (see top of file).


def squarefree_divisors(primes: List[int], N: int) -> List[int]:
    """
    All squarefree divisors P of N built from 'primes',
    excluding 1 and N.
    """
    divisors: Set[int] = set()
    k = len(primes)
    for r in range(1, k + 1):
        for comb in itertools.combinations(primes, r):
            P = 1
            for p in comb:
                P *= p
            if 1 < P < N:
                divisors.add(P)
    return sorted(divisors)


# -------------------------------------------------------
# Orbit logic
# -------------------------------------------------------

def orbit_data_for_P(N: int, P: int) -> Dict[str, Any]:
    """
    For fixed N and squarefree divisor P:

      base = N / P
      centers: C = A * base  (A >= 1, gcd(A, P) = 1),
               only those that produce some |C ± P**x| <= N
      exponents: x >= 1 that actually appear
      values: distinct |C ± P**x| in [1, N]

    Returns a dict with keys:
      "P", "base", "exponents", "centers", "values"
    """
    base = N // P
    values: Set[int] = set()
    centers: Set[int] = set()
    exponents: Set[int] = set()

    # crude upper bound on A so some |C ± P| <= N
    maxA = (N + P) // base
    if maxA < 1:
        return {"P": P, "base": base, "exponents": [], "centers": [], "values": []}

    for A in range(1, maxA + 1):
        if math.gcd(A, P) != 1:
            continue

        C = A * base
        used_center = False
        x = 1
        while True:
            step = P ** x
            v_plus = C + step
            v_minus = C - step

            keep_plus = abs(v_plus) <= N
            keep_minus = abs(v_minus) <= N

            if keep_plus:
                values.add(abs(v_plus))
                exponents.add(x)
                used_center = True
            if keep_minus:
                values.add(abs(v_minus))
                exponents.add(x)
                used_center = True

            # once both branches are outside [-N, N], higher x only gets worse
            if not keep_plus and not keep_minus:
                break
            x += 1

        if used_center:
            centers.add(C)

    return {
        "P": P,
        "base": base,
        "exponents": sorted(exponents),
        "centers": sorted(centers),
        "values": sorted(v for v in values if 1 <= v <= N),
    }


def orbit_summary_for_primorial(primes: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Compute orbit data for all squarefree divisors P of N = primorial(primes).

    Prints a summary (like the N=30 / N=210 examples) and returns
    a dict mapping P -> orbit data.
    """
    N = primorial(primes)
    print(f"N = {N}  (primorial of primes {primes})\n")

    Ps = squarefree_divisors(primes, N)
    summary: Dict[int, Dict[str, Any]] = {}

    for P in Ps:
        data = orbit_data_for_P(N, P)
        summary[P] = data

        print(f"### P = {P}")
        print(f"* base = {N}/{P} = {data['base']}")
        print(f"* exponents used: x = {data['exponents']}")
        print(f"* centers (count {len(data['centers'])}):")
        print(f"  {data['centers'][0:100]}")
        print(f"* values in [1, {N}] (count {len(data['values'])}):")
        print(f"  {data['values'][0:100]}\n")

    # global check against the reduced residue system
    all_values: Set[int] = set()
    for data in summary.values():
        all_values.update(data["values"])

    rr_system = {k for k in range(1, N) if math.gcd(k, N) == 1}

    print("=== Global check ===")
    print(f"Total distinct values from all orbits: {len(all_values)}")
    print(f"φ(N) (coprimes in [1,N)):            {len(rr_system)}")
    print(f"Missing from orbits: {len(rr_system - all_values)}")
    print(f"Extra (non-coprimes): {len(all_values - rr_system)}")

    return summary


def orbit_summary_for_max_prime(max_prime: int) -> Dict[int, Dict[str, Any]]:
    """
    Convenience wrapper:

    - generate all primes <= max_prime,
    - build the primorial N,
    - run the orbit summary.

    Returns the same summary dict as orbit_summary_for_primorial.
    """
    primes = primes_up_to(max_prime)
    if not primes:
        raise ValueError("No primes found up to max_prime")
    return orbit_summary_for_primorial(primes)


# -------------------------------------------------------
# Optional plotting (you can ignore or comment out)
# -------------------------------------------------------

def plot_orbit_values(N: int, summary: Dict[int, Dict[str, Any]]) -> None:
    """
    Scatter plot: x-axis = index of P, y-axis = orbit values in [1,N].
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    Ps = sorted(summary.keys())
    plt.figure()
    for idx, P in enumerate(Ps):
        vals = summary[P]["values"]
        xs = [idx] * len(vals)
        plt.scatter(xs, vals, s=6)

    plt.xlabel("index of squarefree divisor P")
    plt.ylabel(f"values in [1, {N}]")
    plt.title(f"Orbit values per P for N={N}")
    plt.grid(True, alpha=0.3)
    plt.show()


# -------------------------------------------------------
# Run inside PyCharm by just hitting "Run"
# -------------------------------------------------------

if __name__ == "__main__":
    # Choose ANY prime set you want here.
    # Examples:
    #   [2, 3, 5, 7, 11]         -> 2310
    #   [2, 5, 11, 17]          -> 1870
    #   [3, 11, 19]             -> 627
    primes = [2,3,5,7,11]

    summary = orbit_summary_for_primorial(primes)

    # If you want a plot, uncomment:
    # N = primorial(primes)
    # plot_orbit_values(N, summary)
