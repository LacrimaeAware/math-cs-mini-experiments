"""
prime_squares_distance_log_terms.py — how often q^2 - p^2 is divisible by 5.

For consecutive primes p, q, count the fraction of differences q^2 - p^2 that are
divisible by 5, reporting the running percentage at doubling checkpoints. Built
to scale to ~100M primes if the input file and RAM allow.

REQUIRES an input file `primes.txt` (one prime per line) in the working
directory — it is NOT included in this repo. Generate or download one, or point
PRIME_FILE (in the CONFIG block below) at your own list.
"""

import pandas as pd
import math

# ====== CONFIG ======
PRIME_FILE = "primes.txt"   # one prime per line
MAX_PRIMES = 100_000_000    # how many primes to read from file
SHOW_SAMPLE_POINTS = True   # whether to print intermediate stats
# ====================

def load_primes(path, n):
    with open(path, "r") as f:
        primes = []
        for i, line in enumerate(f):
            if i >= n:
                break
            primes.append(int(line.strip()))
    return primes

def auto_checkpoints(n, start=50):
    cps = []
    k = start
    while k < n:
        cps.append(k)
        k *= 2
    if cps[-1] != n:
        cps.append(n)
    return cps

def compute_factor5_stats(primes, checkpoints):
    results = []
    count_with_5 = 0
    for i in range(1, len(primes)):
        p, q = primes[i-1], primes[i]
        diff = q*q - p*p
        if diff % 5 == 0:
            count_with_5 += 1
        if SHOW_SAMPLE_POINTS and (i+1) in checkpoints:
            percent = 100.0 * count_with_5 / i
            results.append((i+1, percent))
    total_percent = 100.0 * count_with_5 / (len(primes) - 1)
    return results, total_percent

def main():
    print(f"Loading up to {MAX_PRIMES:,} primes from file...")
    primes = load_primes(PRIME_FILE, MAX_PRIMES)
    print(f"Loaded {len(primes):,} primes.")

    checkpoints = auto_checkpoints(len(primes))
    stats, total_percent = compute_factor5_stats(primes, checkpoints)

    if SHOW_SAMPLE_POINTS:
        df = pd.DataFrame(stats, columns=["number_of_primes", "percent_with_factor_5"])
        print(df.to_string(index=False))

    print(f"\nFinal % of consecutive prime-square differences divisible by 5: {total_percent:.4f}%")

if __name__ == "__main__":
    main()
