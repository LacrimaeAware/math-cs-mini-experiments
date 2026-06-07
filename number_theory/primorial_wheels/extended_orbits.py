#!/usr/bin/env python3
"""
extended_orbits.py — "extended orbit" coprime construction vs Euler's phi.

A richer cousin of orbit_primorials.py: for N = primorial(primes), sweep all
squarefree divisors P and centers C = A*(N/P) (gcd(A, P) = 1), collect every
|C +/- P^x| <= N, then compare that set against the brute-force coprime set (or
just phi(N) for large N). Prints how many coprimes are covered, missing, or
spuriously produced. Several N configurations are set up in __main__.
"""

import math
import itertools

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


def squarefree_divisors(primes, N):
    """
    All squarefree divisors P of N built from 'primes',
    excluding 1 and N.
    """
    divisors = set()
    for r in range(1, len(primes) + 1):
        for comb in itertools.combinations(primes, r):
            P = 1
            for p in comb:
                P *= p
            if 1 < P < N:
                divisors.add(P)
    return sorted(divisors)


def extended_orbit_values(N, primes, use_progress=True):
    """
    Extended orbit scheme:

      N is the primorial of 'primes'.

      For each squarefree divisor P of N (1 < P < N),
      for each integer A >= 1 with gcd(A, P) = 1 such that
          C = A * (N / P)
      can still produce |C ± P^x| <= N for some x >= 1,
      generate all values:

          |C + P^x|, |C - P^x|

      that satisfy |value| <= N, for all x until both
      branches are out of range.

    Return the full set of absolute values produced.
    """
    divisors = squarefree_divisors(primes, N)
    values = set()

    if use_progress and HAVE_TQDM:
        div_iter = tqdm(divisors, desc="Divisors P", unit="P")
    else:
        div_iter = divisors

    for P in div_iter:
        base = N // P  # N / P is integer since P divides N

        # A bound: need at least |A*base| - P <= N to have any x with |C ± P^x| <= N
        # => |A*base| <= N + P  =>  A <= (N + P) / base for A >= 1.
        maxA = (N + P) // base
        if maxA < 1:
            continue

        for A in range(1, int(maxA) + 1):
            if math.gcd(A, P) != 1:
                continue

            C = A * base
            x = 1
            while True:
                step = P ** x
                v_plus = C + step
                v_minus = C - step

                keep_plus = abs(v_plus) <= N
                keep_minus = abs(v_minus) <= N

                if keep_plus:
                    values.add(abs(v_plus))
                if keep_minus:
                    values.add(abs(v_minus))

                if not keep_plus and not keep_minus:
                    break  # larger x only increases |step|

                x += 1

    return values


def traditional_coprimes_set(N):
    """
    Traditional construction: all residues 1 <= k < N with gcd(k, N) = 1.
    """
    return {k for k in range(1, N) if math.gcd(k, N) == 1}


def phi_from_primes(N, primes):
    """
    Compute Euler's totient φ(N) multiplicatively from the prime factors.
    Assumes 'primes' multiply to N.
    """
    phi = N
    for p in primes:
        phi = phi // p * (p - 1)
    return phi


def analyze_configuration(N, primes, print_limit=50,
                          build_traditional=True, use_progress=True):
    """
    Run the extended-orbit construction for given N, primes, and compare
    against the traditionally constructed coprime set.

    - print_limit: how many integers to print from each set (to avoid huge dumps).
    - build_traditional: if False, only φ(N) is computed multiplicatively.
    """
    print("=" * 80)
    print(f"Analyzing N = {N}")
    print(f"Primes = {primes}")
    print("=" * 80)

    # Extended-orbit values
    orbit_vals = extended_orbit_values(N, primes, use_progress=use_progress)

    # Split into coprimes and non-coprimes explicitly
    orbit_coprimes = {v for v in orbit_vals if 1 <= v < N and math.gcd(v, N) == 1}
    orbit_noncoprimes = {v for v in orbit_vals if not (1 <= v < N and math.gcd(v, N) == 1)}

    print(f"\nExtended-orbit output:")
    print(f"  Total distinct absolute values in [1, N]: {len(orbit_vals)}")
    print(f"  Of those, coprime to N: {len(orbit_coprimes)}")
    print(f"  Of those, NON-coprime to N (ideally 0): {len(orbit_noncoprimes)}")

    if orbit_noncoprimes:
        print(f"  Example non-coprimes: {sorted(orbit_noncoprimes)[:print_limit]}")

    # Traditional coprimes / φ(N)
    if build_traditional:
        print("\nBuilding traditional coprime set (brute force)…")
        trad_coprimes = traditional_coprimes_set(N)
        phi_val = len(trad_coprimes)
    else:
        trad_coprimes = None
        phi_val = phi_from_primes(N, primes)

    print(f"\nEuler φ(N) = {phi_val}")

    if trad_coprimes is not None:
        # Compare sets
        missing_from_orbit = trad_coprimes - orbit_coprimes
        extra_in_orbit = orbit_coprimes - trad_coprimes

        print(f"\nComparison:")
        print(f"  Orbit coprimes count: {len(orbit_coprimes)}")
        print(f"  Traditional coprimes count: {len(trad_coprimes)}")
        print(f"  Missing from orbit (trad − orbit): {len(missing_from_orbit)}")
        print(f"  Extra in orbit (orbit − trad): {len(extra_in_orbit)}")

        if missing_from_orbit:
            print(f"  Example missing: {sorted(missing_from_orbit)[:print_limit]}")
        if extra_in_orbit:
            print(f"  Example extra: {sorted(extra_in_orbit)[:print_limit]}")
    else:
        print(f"\nNo explicit traditional set was built.")
        print(f"  Orbit coprimes count: {len(orbit_coprimes)}")
        print(f"  φ(N) (target count):  {phi_val}")
        print(f"  Difference (orbit − φ): {len(orbit_coprimes) - phi_val}")

    print("\nSample of coprimes found by orbit construction:")
    sorted_orbit_coprimes = sorted(orbit_coprimes)
    print(sorted_orbit_coprimes[:print_limit])
    if len(sorted_orbit_coprimes) > print_limit:
        print(f"... (total {len(sorted_orbit_coprimes)} values)")

    print("=" * 80)
    print()


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # CONFIG 1: N = 2 * 3 * 5 * 7 * 11 = 2310
    # ------------------------------------------------------------------
    primes_11 = [2, 3, 5, 7, 11]
    N_11 = math.prod(primes_11)

    analyze_configuration(
        N=N_11,
        primes=primes_11,
        print_limit=50,
        build_traditional=True,   # small enough to brute force
        use_progress=True,
    )

    # ------------------------------------------------------------------
    # CONFIG 2: N = 2 * 3 * 5 * 7 * 11 * 13 * 17 = 510510
    #          φ(N) = 92160   (easy)
    # ------------------------------------------------------------------
    primes_17 = [2, 3, 5, 7, 11, 13, 17]
    N_17 = math.prod(primes_17)

    analyze_configuration(
        N=N_17,
        primes=primes_17,
        print_limit=50,
        build_traditional=True,   # still fine to brute force
        use_progress=True,
    )

    # ------------------------------------------------------------------
    # CONFIG 3: N = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 = 223092870
    #          φ(N) = 36495360   (≈ 36 million)
    #
    # WARNING:
    #   Building the full traditional coprime set for this N means
    #   iterating up to 223M and storing ~36M integers. That is doable
    #   but slow and memory-heavy.
    #
    # By default, we only compare counts using φ(N). If you really want
    # the full set comparison, set build_traditional=True (expensive).
    # ------------------------------------------------------------------
    primes_23 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_23 = math.prod(primes_23)

    analyze_configuration(
        N=N_23,
        primes=primes_23,
        print_limit=50,
        build_traditional=False,  # set to True only for large runs only
        use_progress=True,
    )

    # ------------------------------------------------------------------
    # CONFIG 4 (NOT RUN BY DEFAULT): primes up to 29
    #   N would be about 6.47e9, φ(N) about ~1e9 level.
    #   Not practical to fully enumerate coprimes or orbit output.
    # ------------------------------------------------------------------
    # primes_29 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    # N_29 = math.prod(primes_29)
    # analyze_configuration(
    #     N=N_29,
    #     primes=primes_29,
    #     print_limit=50,
    #     build_traditional=False,
    #     use_progress=True,
    # )
