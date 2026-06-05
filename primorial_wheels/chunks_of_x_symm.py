"""
chunks_of_x_symm.py — coprime counts per block of a primorial wheel.

For N = primorial of all primes <= --pmax, partition [1..limit] into fixed-size
blocks (--block) and print the number of integers coprime to N in each block.
A tool for spotting symmetry/structure in how a primorial's totatives spread
across equal-length windows. Example:

    python chunks_of_x_symm.py --pmax 19 --block 510510
"""

import math
import argparse
from typing import List, Tuple

# Project-local shared helpers (see prime_lib.py at repo root).
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from prime_lib import primes_up_to

# primorial() below intentionally stays local: it returns (N, primes), unlike
# the prime_lib version which returns just N.

def primorial(p_max: int) -> Tuple[int, List[int]]:
    ps = primes_up_to(p_max)
    N = 1
    for p in ps:
        N *= p
    return N, ps

def coprime_counts_by_block(N: int, block_len: int, limit: int = None):
    if limit is None:
        limit = N
    counts = []
    lengths = []

    start = 1
    while start <= limit:
        end = min(start + block_len - 1, limit)
        length = end - start + 1
        c = 0
        for n in range(start, end + 1):
            if math.gcd(n, N) == 1:
                c += 1
        counts.append(c)
        lengths.append(length)
        start = end + 1

    return counts, lengths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmax", type=int, default=19,
                    help="largest prime in primorial modulus")
    ap.add_argument("--block", type=int, default=int(round(2*3*5*7*11*13*17)),
                    help="block length")
    ap.add_argument("--limit", type=int, default=None,
                    help="analyze [1..limit] instead of [1..N]")
    args = ap.parse_args()

    N, _ = primorial(args.pmax)
    counts, lengths = coprime_counts_by_block(N, args.block, args.limit)

    # print(lengths)  # row of block lengths
    print(counts)   # single array of coprime counts per block

if __name__ == "__main__":
    main()
