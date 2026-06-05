# head_tail.py
#
# Scan N = A * base**k + X over grids in (A, k) for given tails X and bases,
# record primality, compare to expected prime counts (sum 1/log N),
# and save BOTH:
#   - heatmaps (optional)
#   - raw data + summary stats to files
#
# Bud knobs:
#   - BASE_LIST, X_LIST
#   - A_MIN / A_MAX or A_LIST
#   - K_MIN / K_MAX or K_LIST

import math
import os
import random
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# Project-local shared helpers + output directory (see prime_lib.py at repo root).
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from prime_lib import is_probable_prime, OUTPUTS


# -------------------------
# CONFIGURATION
# -------------------------

BASE_LIST = [10]       # e.g. [10], [2, 10], [2, 3, 5, 10]

X_LIST = [997]

# Ranges for A, k (used if USE_*_LIST is False)
A_MIN, A_MAX = 19, 19
K_MIN, K_MAX = 1, 1500

# Explicit lists (set USE_*_LIST = True to enable)
USE_A_LIST = False
A_LIST = [19, 7, 6]

USE_K_LIST = False
K_LIST = [1, 2, 3, 10]

MR_ROUNDS = 16

MAKE_HEATMAPS = True
HEATMAP_DIR = str(OUTPUTS / "heatmaps")   # all runs write here
DATA_DIR = str(OUTPUTS / "run_data")      # <- raw data goes here


# -------------------------
# PRIMALITY TEST
# -------------------------

# is_probable_prime() is imported from prime_lib (see top of file).


# -------------------------
# VALUE GENERATORS
# -------------------------

def get_A_values() -> list[int]:
    if USE_A_LIST:
        return sorted(set(A_LIST))
    return list(range(A_MIN, A_MAX + 1))


def get_K_values() -> list[int]:
    if USE_K_LIST:
        return sorted(set(K_LIST))
    return list(range(K_MIN, K_MAX + 1))


# -------------------------
# PROGRESS BAR
# -------------------------

def progress_bar(current: int, total: int, prefix: str = ""):
    if total <= 0:
        return
    frac = current / total
    percent = int(frac * 100)
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    msg = f"\r{prefix} [{bar}] {percent:3d}% ({current}/{total})"
    print(msg, end="", flush=True)
    if current == total:
        print()  # newline


# -------------------------
# GRID SCAN
# -------------------------

def scan_grid_for_tail(X: int,
                       base: int,
                       A_vals: list[int],
                       K_vals: list[int]):
    """
    Returns (grid, stats) where:
      grid[i][j] = 1 if A_vals[i]*base**K_vals[j] + X is prime, else 0
      stats = dict with:
        - total_points
        - prime_count
        - expected_total
        - primes_per_A
        - expected_per_A
        - primes_per_k
    """
    grid = [[0 for _ in K_vals] for __ in A_vals]

    total_points = len(A_vals) * len(K_vals)
    done = 0

    prime_count = 0
    expected_total = 0.0

    primes_per_A = Counter()
    primes_per_k = Counter()
    expected_per_A = Counter()

    for i, A in enumerate(A_vals):
        for j, k in enumerate(K_vals):
            N = A * (base ** k) + X

            if N > 1:
                w = 1.0 / math.log(N)
            else:
                w = 0.0
            expected_total += w
            expected_per_A[A] += w

            if is_probable_prime(N):
                grid[i][j] = 1
                prime_count += 1
                primes_per_A[A] += 1
                primes_per_k[k] += 1

            done += 1
            step = max(total_points // 100, 1000)
            if done == 1 or done % step == 0 or done == total_points:
                progress_bar(done, total_points,
                             prefix=f"  scanning A*{base}^k + {X}")

    stats = {
        "A_vals": A_vals,
        "K_vals": K_vals,
        "total_points": total_points,
        "prime_count": prime_count,
        "expected_total": expected_total,
        "primes_per_A": primes_per_A,
        "expected_per_A": expected_per_A,
        "primes_per_k": primes_per_k,
    }
    return grid, stats


# -------------------------
# HEATMAP
# -------------------------

def plot_heatmap(grid, A_vals, K_vals, X: int, base: int,
                 A_range: tuple[int, int], K_range: tuple[int, int]):
    os.makedirs(HEATMAP_DIR, exist_ok=True)

    data = np.array(grid, dtype=float)

    plt.figure(figsize=(8, 6))
    plt.imshow(data, aspect='auto', origin='lower')
    plt.colorbar(label='prime (1) / composite (0)')

    plt.xlabel("k (exponent)")
    plt.ylabel("A (head coefficient)")
    plt.xticks(
        ticks=range(0, len(K_vals), max(1, len(K_vals) // 10)),
        labels=[str(K_vals[i]) for i in range(0, len(K_vals), max(1, len(K_vals) // 10))]
    )
    plt.yticks(
        ticks=range(0, len(A_vals), max(1, len(A_vals) // 10)),
        labels=[str(A_vals[i]) for i in range(0, len(A_vals), max(1, len(A_vals) // 10))]
    )
    plt.title(f"N = A*{base}^k + {X}")

    plt.tight_layout()

    A_lo, A_hi = A_range
    K_lo, K_hi = K_range
    fname = f"heat_base{base}_X{X}_A{A_lo}-{A_hi}_k{K_lo}-{K_hi}.png"
    path = os.path.join(HEATMAP_DIR, fname)

    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved heatmap to {path}")


# -------------------------
# DATA SAVE
# -------------------------

def save_run_data(grid,
                  stats,
                  base: int,
                  X: int,
                  A_range: tuple[int, int],
                  K_range: tuple[int, int]):
    """
    Save:
      - CSV with row per (A, k): A,k,N,is_prime
      - JSON with summary stats (counts and expectations)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    A_vals = stats["A_vals"]
    K_vals = stats["K_vals"]
    A_lo, A_hi = A_range
    K_lo, K_hi = K_range

    # --- CSV of pointwise data ---
    csv_name = f"data_base{base}_X{X}_A{A_lo}-{A_hi}_k{K_lo}-{K_hi}.csv"
    csv_path = os.path.join(DATA_DIR, csv_name)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("A,k,N,is_prime\n")
        for i, A in enumerate(A_vals):
            for j, k in enumerate(K_vals):
                N = A * (base ** k) + X
                is_prime_flag = grid[i][j]
                f.write(f"{A},{k},{N},{is_prime_flag}\n")

    print(f"  saved data to {csv_path}")

    # --- JSON summary ---
    json_name = f"summary_base{base}_X{X}_A{A_lo}-{A_hi}_k{K_lo}-{K_hi}.json"
    json_path = os.path.join(DATA_DIR, json_name)

    # Convert Counters to plain dicts for JSON
    primes_per_A = {int(a): int(c) for a, c in stats["primes_per_A"].items()}
    expected_per_A = {int(a): float(e) for a, e in stats["expected_per_A"].items()}
    primes_per_k = {int(k): int(c) for k, c in stats["primes_per_k"].items()}

    summary = {
        "base": base,
        "X": X,
        "A_range": [A_lo, A_hi],
        "K_range": [K_lo, K_hi],
        "total_points": int(stats["total_points"]),
        "prime_count": int(stats["prime_count"]),
        "expected_total": float(stats["expected_total"]),
        "primes_per_A": primes_per_A,
        "expected_per_A": expected_per_A,
        "primes_per_k": primes_per_k,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  saved summary to {json_path}")


# -------------------------
# MAIN
# -------------------------

def main():
    A_vals = get_A_values()
    K_vals = get_K_values()
    A_lo, A_hi = A_vals[0], A_vals[-1]
    K_lo, K_hi = K_vals[0], K_vals[-1]

    print(f"A in [{A_lo},{A_hi}] ({len(A_vals)} values)")
    print(f"k in [{K_lo},{K_hi}] ({len(K_vals)} values)\n")

    for base in BASE_LIST:
        if base <= 1:
            print(f"WARNING: base = {base} is degenerate "
                  f"(N = A*{base}^k + X collapses / behaves weirdly).")

        for X in X_LIST:
            print(f"=== base = {base}, tail X = {X} ===")

            grid, stats = scan_grid_for_tail(X, base, A_vals, K_vals)

            total = stats["total_points"]
            prime_count = stats["prime_count"]
            expected_total = stats["expected_total"]

            actual_density = prime_count / total if total > 0 else 0.0
            expected_density = expected_total / total if total > 0 else 0.0

            print(f"Total points: {total}")
            print(f"Primes: {prime_count}  (actual density = {actual_density:.6g})")
            print(f"Expected primes (sum 1/log N): {expected_total:.6g}")
            print(f"Expected density: {expected_density:.6g}")
            if expected_total > 0:
                print(f"Global ratio actual/expected ≈ {prime_count / expected_total:.3f}")
            print()

            # per-A stats (stdout)
            per_A = stats["primes_per_A"]
            exp_A = stats["expected_per_A"]

            if per_A:
                top_A_by_count = sorted(per_A.items(), key=lambda kv: kv[1], reverse=True)[:10]
                print("Top A by prime count (A: count):")
                print(", ".join(f"{a}:{c}" for a, c in top_A_by_count))
                print()

                scores = []
                for a, c in per_A.items():
                    ea = exp_A[a]
                    if ea > 0:
                        scores.append((a, c / ea))
                scores.sort(key=lambda kv: kv[1], reverse=True)
                top_A_by_score = scores[:10]

                print("Top A by score(A) = count / expected (A: score):")
                print(", ".join(f"{a}:{s:.2f}" for a, s in top_A_by_score))
                print()

            per_k = stats["primes_per_k"]
            if per_k:
                top_k = sorted(per_k.items(), key=lambda kv: kv[1], reverse=True)[:10]
                print("Top k by prime count (k: count):")
                print(", ".join(f"{k}:{c}" for k, c in top_k))
                print()

            # save raw data + summary
            save_run_data(grid, stats, base, X, (A_lo, A_hi), (K_lo, K_hi))

            # optional heatmap
            if MAKE_HEATMAPS:
                plot_heatmap(grid, A_vals, K_vals, X, base, (A_lo, A_hi), (K_lo, K_hi))

    print("Done.")


if __name__ == "__main__":
    main()
