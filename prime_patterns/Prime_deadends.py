"""
Prime_deadends.py — search for left-truncatable prime chains by prepending digits.

Starting from a fixed TAIL, depth-first prepend digits (1..9 by default) to the
front, keeping only branches that stay prime (Miller-Rabin). Records the deepest
chain reached and the "dead ends" where no prime extension exists. Optionally
first verifies that TAIL itself is left-truncatable prime. Writes a summary to
outputs/prefix_chain_summary.txt.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

# Project-local shared helpers + output directory (see prime_lib.py at repo root).
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from prime_lib import is_probable_prime, ensure_output_dir

# ==============================
# CONFIG
# ==============================

# Tail T: the fixed suffix. Example:
#  - 3212336353  (your left-truncatable prime tail)
#  - 997         (simpler test)
TAIL = 56357686312646216567629137

# Maximum number of prepended digits (depth of search tree)
# depth = 0 means just TAIL
# depth = 1 means d TAIL  (d in DIGITS)
# depth = 2 means d1 d2 TAIL, etc.
MAX_DEPTH = 1000

# Which digits are allowed to prepend.
# For "no leading zero", keep 1..9; if you want internal zeros as you go deeper,
# just leave 0 in here – the code doesn’t special-case depth.
DIGITS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Progress printing: print status every N visited nodes
PROGRESS_EVERY = 1000

# Output file for summary
SUMMARY_FILE = str(ensure_output_dir() / "prefix_chain_summary.txt")

# Miller–Rabin rounds
MR_ROUNDS = 16


# ==============================
# PRIMALITY TEST
# ==============================

# is_probable_prime() is imported from prime_lib (see top of file).


# ==============================
# UTILITIES
# ==============================

def num_digits(n: int) -> int:
    """Number of decimal digits of |n|."""
    return len(str(abs(n)))


def prepend_digit(d: int, n: int) -> int:
    """Return integer for decimal concatenation: d || n."""
    return d * (10 ** num_digits(n)) + n


def left_truncations(n: int) -> List[int]:
    """
    Return all left truncations of n, including n itself, down to the last digit.
    e.g. 3212336353 -> [3212336353, 212336353, 12336353, 2336353, ...]
    """
    s = str(abs(n))
    return [int(s[i:]) for i in range(len(s))]


def check_left_truncatable_prime_tail(tail: int) -> bool:
    """
    Check that tail and all its left truncations are prime.
    This is what you described for 3212336353.
    """
    for x in left_truncations(tail):
        if not is_probable_prime(x):
            return False
    return True


# ==============================
# SEARCH STATE
# ==============================

@dataclass
class SearchStats:
    visited: int = 0
    dead_ends: int = 0
    best_depth: int = 0
    best_chain_digits: List[int] = field(default_factory=list)
    best_number: Optional[int] = None
    dead_end_examples: List[int] = field(default_factory=list)


# ==============================
# DIGIT ORDER STRATEGY
# ==============================

def digit_order(last_digit: Optional[int]) -> List[int]:
    """
    Choose the order in which to try digits at this step.

    Right now: simple fixed order.
    You can modify this to implement your "reverse based on last digit" idea.

    Example for your idea (commented out):
        - if last_digit is None: try 1..9
        - if last_digit in {1,2,3,4,5}: try 9..1
        - else: try 1..9

    For now, we return DIGITS unchanged.
    """
    # --- simple version ---
    return DIGITS

    # --- example variant you can experiment with ---
    # if last_digit is None:
    #     return DIGITS[:]           # 1..9
    # if last_digit <= 5:
    #     return sorted(DIGITS, reverse=True)  # 9..1
    # else:
    #     return DIGITS[:]           # 1..9


# ==============================
# DFS OVER PREFIX CHAINS
# ==============================

def dfs_prefix_chain(current_n: int,
                     depth: int,
                     chain_digits: List[int],
                     stats: SearchStats):
    """
    Depth-first search over all ways to prepend digits to current_n,
    up to MAX_DEPTH. Records dead ends and longest chain.
    """

    stats.visited += 1

    # progress printing
    if stats.visited % PROGRESS_EVERY == 0:
        print(
            f"[progress] visited={stats.visited}, depth={depth}, "
            f"best_depth={stats.best_depth}, current={current_n}"
        )

    # update best chain
    if depth > stats.best_depth:
        stats.best_depth = depth
        stats.best_chain_digits = chain_digits.copy()
        stats.best_number = current_n
        print(f"[new best] depth={depth}, number={current_n}, digits={chain_digits}")

    # stop if we hit max depth
    if depth >= MAX_DEPTH:
        return

    # try to extend by prepending digits
    children_found = False
    last_digit = chain_digits[-1] if chain_digits else None
    for d in digit_order(last_digit):
        new_n = prepend_digit(d, current_n)
        if is_probable_prime(new_n):
            children_found = True
            new_chain = chain_digits + [d]
            dfs_prefix_chain(new_n, depth + 1, new_chain, stats)

    # if no prime children, this is a dead end
    if not children_found:
        stats.dead_ends += 1
        stats.dead_end_examples.append(current_n)
        print(f"[dead end] depth={depth}, number={current_n}, digits={chain_digits}")


# ==============================
# MAIN
# ==============================

def main():
    print(f"Tail T = {TAIL}")
    print(f"MAX_DEPTH = {MAX_DEPTH}")
    print(f"Allowed digits to prepend: {DIGITS}")
    print()

    # Optional: verify "left-truncatable prime tail" property
    print("Checking whether tail is left-truncatable prime...")
    if check_left_truncatable_prime_tail(TAIL):
        print("  Tail IS left-truncatable prime (all left truncations are prime).")
    else:
        print("  Tail is NOT left-truncatable prime (some truncation is composite).")
    print()

    stats = SearchStats()

    try:
        # depth=0, chain_digits=[], start at TAIL
        dfs_prefix_chain(current_n=TAIL,
                         depth=0,
                         chain_digits=[],
                         stats=stats)
    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] Stopping search early...")

    print("\n=== SEARCH SUMMARY ===")
    print(f"Total nodes visited: {stats.visited}")
    print(f"Total dead ends:    {stats.dead_ends}")
    print(f"Best depth:         {stats.best_depth}")
    if stats.best_number is not None:
        print(f"Best number:        {stats.best_number}")
        print(f"Digits prepended:   {stats.best_chain_digits}")
    else:
        print("No extensions found (tail itself had no prime extensions).")

    # save summary + some dead-end examples to file
    try:
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write(f"Tail T = {TAIL}\n")
            f.write(f"MAX_DEPTH = {MAX_DEPTH}\n")
            f.write(f"Digits used = {DIGITS}\n\n")
            f.write(f"Total nodes visited: {stats.visited}\n")
            f.write(f"Total dead ends:    {stats.dead_ends}\n")
            f.write(f"Best depth:         {stats.best_depth}\n")
            if stats.best_number is not None:
                f.write(f"Best number:        {stats.best_number}\n")
                f.write(f"Digits prepended:   {stats.best_chain_digits}\n")
            f.write("\nDead-end examples (up to first 20):\n")
            for n in stats.dead_end_examples[:20]:
                f.write(str(n) + "\n")
        print(f"\nSummary written to {SUMMARY_FILE}")
    except Exception as e:
        print(f"\n[warning] Failed to write summary: {e}")


if __name__ == "__main__":
    main()
