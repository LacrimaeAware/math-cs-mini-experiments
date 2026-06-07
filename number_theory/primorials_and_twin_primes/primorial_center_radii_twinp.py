"""
primorial_center_radii_twinp.py — exact L/R expressions around P/A^2.

For each prime cutoff p_max (giving primorial P) and each (A, B): compute the
exact midpoint m = nearest_int(P / A^2), set L = m - B and R = m + B, and
evaluate the exact rational expressions A*R - P/A and -A*L + P/A. Optionally
factors L, R and m (sympy if installed, else trial division). All arithmetic is
exact (Fractions / ints) — no floats.
"""

from typing import List, Dict, Any
from fractions import Fraction

# Project-local shared sieve (see prime_lib.py at repo root).
import sys
import pathlib
sys.path.append(next(str(p) for p in pathlib.Path(__file__).resolve().parents if (p / "prime_lib.py").exists()))
from prime_lib import primes_up_to

# ============================================================
# CONFIG: edit these and run
# ============================================================

P_MAXES = [43]  # list of prime cutoffs to test
AS = [58]                          # default A = 6
BS = [58]                         # default B = 17

DO_FACTOR_LR = True               # factor L and R by default
FACTOR_MIDPOINT = True            # factor m by default (only if DO_FACTOR_LR True)

# ============================================================
# Prime / primorial utilities
# ============================================================

# primes_up_to() is imported from prime_lib (see top of file).

def primorial_up_to_prime(p_max: int) -> int:
    ps = primes_up_to(p_max)
    P = 1
    for q in ps:
        P *= q
    return P

def nearest_int_fraction(x: Fraction) -> int:
    """
    Exact nearest-integer to a Fraction.
    Ties go to the nearest even integer (like Python round()).
    """
    n = x.numerator
    d = x.denominator
    q, r = divmod(n, d)  # floor and remainder

    if 2 * r < d:
        return q
    if 2 * r > d:
        return q + 1

    # exact tie -> round to even
    return q if (q % 2 == 0) else q + 1

# ============================================================
# Factoring (optional, exact inputs)
# ============================================================

def trial_factorization(x: int) -> Dict[int, int]:
    factors: Dict[int, int] = {}
    if x == 0:
        return {0: 1}
    n = abs(x)
    if n == 1:
        return {1: 1}

    while n % 2 == 0:
        factors[2] = factors.get(2, 0) + 1
        n //= 2

    f = 3
    while f * f <= n:
        while n % f == 0:
            factors[f] = factors.get(f, 0) + 1
            n //= f
        f += 2

    if n > 1:
        factors[n] = factors.get(n, 0) + 1

    return factors

try:
    from sympy import factorint as sympy_factorint
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False
    sympy_factorint = None

def prime_factorization(x: int) -> Dict[int, int]:
    if HAVE_SYMPY:
        return dict(sympy_factorint(x))
    return trial_factorization(x)

def fmt_factors(x: int) -> str:
    fac = prime_factorization(x)
    if fac == {0: 1}:
        return "0"
    if fac == {1: 1}:
        return "1"
    parts = []
    if x < 0:
        parts.append("-1")
    for p in sorted(k for k in fac if k not in (0, 1)):
        e = fac[p]
        parts.append(f"{p}^{e}" if e > 1 else str(p))
    return " * ".join(parts)

# ============================================================
# Core scan (NO FLOATS)
# ============================================================

def run_scan_for_pmax(
    p_max: int,
    As: List[int],
    Bs: List[int],
    do_factor_LR: bool = False,
    factor_midpoint: bool = False,
) -> List[Dict[str, Any]]:
    """
    For each A,B with fixed p_max:
      P = primorial up to p_max
      m = nearest_int(P / A^2)   [exact]
      L = m - B
      R = m + B
      right_expr = A*R - P/A     [exact Fraction]
      left_expr  = -A*L + P/A    [exact Fraction]
    """
    P = primorial_up_to_prime(p_max)
    results = []

    for A in As:
        if A == 0:
            raise ValueError("A cannot be 0.")

        t = Fraction(P, A * A)       # exact P/A^2
        m = nearest_int_fraction(t)  # exact nearest integer
        P_over_A = Fraction(P, A)    # exact P/A

        for B in Bs:
            L = m - B
            R = m + B

            right_expr = A * R - P_over_A
            left_expr  = (-A) * L + P_over_A

            row = {
                "p_max": p_max,
                "P_digits": len(str(P)),
                "A": A,
                "B": B,
                "m": m,
                "L": L,
                "R": R,
                "right_expr": right_expr,
                "left_expr": left_expr,
            }

            if do_factor_LR:
                row["factors_L"] = fmt_factors(L)
                row["factors_R"] = fmt_factors(R)
                if factor_midpoint:
                    row["factors_m"] = fmt_factors(m)

            results.append(row)

    return results

# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    if DO_FACTOR_LR and not HAVE_SYMPY:
        print("Warning: sympy not installed; factoring may be VERY slow.")
        print("Install with: pip install sympy\n")

    for p_max in P_MAXES:
        results = run_scan_for_pmax(
            p_max,
            AS,
            BS,
            do_factor_LR=DO_FACTOR_LR,
            factor_midpoint=FACTOR_MIDPOINT,
        )

        for r in results:
            print("\n" + "-" * 60)
            print(f"p_max={r['p_max']}  A={r['A']}  B={r['B']}")
            print(f"P digits={r['P_digits']}")
            print(f"m={r['m']}")
            print(f"L={r['L']}")
            print(f"R={r['R']}")
            print(f"right_expr=6*R - P/A = {r['right_expr']}")
            print(f"left_expr=-A*L + P/A  = {r['left_expr']}")

            if DO_FACTOR_LR:
                print(f"factors(L): {r['factors_L']}")
                print(f"factors(R): {r['factors_R']}")
                if FACTOR_MIDPOINT:
                    print(f"factors(m): {r['factors_m']}")
