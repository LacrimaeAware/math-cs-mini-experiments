"""
prime_lib.py — shared number-theory helpers for the MathExperimentation project.

Background
----------
Many scripts in this repo independently reimplemented the *same* primality
tests, sieves and primorial helpers (often copy-pasted with small variations).
They now live here once and are imported by the individual experiments, so a
fix or speedup in one place benefits every script.

This module also exposes project paths so every script writes its generated
artifacts into a single, predictable ``<repo>/outputs/`` directory regardless of
where the script is launched from (terminal, PyCharm "Run", etc.).

Nothing here has side effects on import except defining ``ROOT`` / ``OUTPUTS``;
directories are only created when you call :func:`ensure_output_dir`.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# This file lives at the repository root, so its parent IS the repo root.
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


def ensure_output_dir(*parts: str) -> Path:
    """Return ``<repo>/outputs/<parts...>``, creating the directory if needed.

    Examples
    --------
    >>> ensure_output_dir()                 # -> <repo>/outputs
    >>> ensure_output_dir("heatmaps")       # -> <repo>/outputs/heatmaps
    """
    d = OUTPUTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Deterministic primality (exact) — fine for small / medium integers
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Exact deterministic primality test by trial division."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def next_prime(p: int) -> int:
    """Smallest prime strictly greater than ``p`` (deterministic)."""
    n = p + 1
    while not is_prime(n):
        n += 1
    return n


# Some scripts historically called this name; keep it as an alias.
next_prime_after = next_prime


def prev_prime(n: int) -> Optional[int]:
    """Largest prime strictly less than ``n``, or ``None`` if none exists."""
    c = n - 1
    while c >= 2:
        if is_prime(c):
            return c
        c -= 1
    return None


def first_n_primes(k: int) -> List[int]:
    """Return the first ``k`` primes (trial division)."""
    primes: List[int] = []
    n = 2
    while len(primes) < k:
        if is_prime(n):
            primes.append(n)
        n += 1
    return primes


# ---------------------------------------------------------------------------
# Probabilistic primality (Miller–Rabin) — for large integers
# ---------------------------------------------------------------------------

def is_probable_prime(n: int, rounds: int = 16) -> bool:
    """Miller–Rabin probable-prime test with ``rounds`` random bases.

    Probabilistic, but with 16 rounds the error probability is astronomically
    small (< 4**-16) — more than reliable enough for these experiments.
    """
    if n < 2:
        return False

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    # write n - 1 = d * 2^s with d odd
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    for _ in range(rounds):
        a = random.randrange(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Sieves and primorials
# ---------------------------------------------------------------------------

def sieve_primes(limit: int) -> List[int]:
    """All primes ``<= limit`` via a simple Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, math.isqrt(limit) + 1):
        if sieve[p]:
            start = p * p
            sieve[start:limit + 1:p] = b"\x00" * (((limit - start) // p) + 1)
    return [i for i, b in enumerate(sieve) if b]


# Several scripts spell the same operation as ``primes_up_to``.
primes_up_to = sieve_primes


def primorial(primes) -> int:
    """Product of an iterable of primes (the primorial of that set)."""
    n = 1
    for p in primes:
        n *= p
    return n


def primorial_up_to(p_max: int) -> int:
    """Primorial of every prime ``<= p_max`` (i.e. ``2 * 3 * 5 * ... * p_max``)."""
    return primorial(sieve_primes(p_max))
