"""
Chunks_of_6_symm.py — visualize coprime residues of a primorial wheel mod 6.

For a primorial-style modulus N, tag each integer in [1..N] as '.' (not coprime
to N), '1' (coprime and == 1 mod 6) or '5' (coprime and == 5 mod 6), then print
three views: complement-sized blocks, non-overlapping groups of 6, and groups of
6 chunked by complement blocks. A visual aid for the 6k +/- 1 structure of a
primorial's totatives.
"""

import math


def tag(n: int, N: int) -> str:
    """
    Return a single character describing n w.r.t. N and mod 6:
      '.' : not coprime to N
      '1' : coprime and n ≡ 1 (mod 6)
      '5' : coprime and n ≡ 5 (mod 6)
      '?' : coprime but some other residue (shouldn't happen for N with 2 and 3)
    """
    if math.gcd(n, N) != 1:
        return "."
    r = n % 6
    if r == 1:
        return "1"
    if r == 5:
        return "5"
    return "?"


# --------------------------------------
# View 1: blocks of size N/6 (complement)
# --------------------------------------

def view_blocks_of_complement(N: int) -> None:
    """
    Partition [1..N] into 6 blocks of length block_len = N//6.
    For each block, print the pattern of coprimes (tag) across that block.
    """
    block_len = N // 6
    print(f"\n=== View 1: blocks of size {block_len} for N={N} ===")

    for b in range(6):
        start = b * block_len + 1
        end = (b + 1) * block_len
        tags = [tag(n, N) for n in range(start, end + 1)]
        print(f"Block {b+1} [{start}-{end}]:")
        # indices line
        idx_line = " ".join(f"{i:2d}" for i in range(1, block_len + 1))
        tag_line = " ".join(f" {c}" for c in tags)
        print(" pos:", idx_line)
        print(" tag:", tag_line)
        print()


# --------------------------------------
# View 2: non-overlapping groups of 6
# --------------------------------------

def view_groups_of_6(N: int) -> None:
    """
    Partition [1..N] into non-overlapping groups of size 6.
    For each group, print the 6 tags.
    """
    group_size = 6
    num_groups = (N + group_size - 1) // group_size  # last group may be partial

    print(f"\n=== View 2: groups of 6 for N={N} ===")

    for g in range(num_groups):
        start = g * group_size + 1
        end = min((g + 1) * group_size, N)
        tags = [tag(n, N) for n in range(start, end + 1)]
        print(f"Group {g+1:2d} [{start}-{end}]: ", end="")
        print(" ".join(tags))


# --------------------------------------------------
# View 3: groups of 6, chunked by complement blocks
# --------------------------------------------------

def view_groups_of_6_chunked_by_complement(N: int) -> None:
    """
    Same groups-of-6 as View 2, but grouped into chunks according to
    which [k*comp+1 .. (k+1)*comp] interval they lie in, where comp = N//6.

    More concretely, for N=210, comp=35:
      chunk 1: groups whose range is within 1..35
      chunk 2: groups whose range is within 36..70
      ...
    """
    comp = N // 6
    group_size = 6
    num_groups = (N + group_size - 1) // group_size
    num_chunks = (N + comp - 1) // comp

    print(f"\n=== View 3: groups of 6 chunked by blocks of size {comp} for N={N} ===")

    # For each chunk (complement block), collect relevant groups
    for c in range(num_chunks):
        chunk_start = c * comp + 1
        chunk_end = min((c + 1) * comp, N)
        print(f"\n--- Chunk {c+1}: [{chunk_start}-{chunk_end}] ---")

        for g in range(num_groups):
            start = g * group_size + 1
            end = min((g + 1) * group_size, N)

            # group is inside this chunk if its interval is fully contained
            if start >= chunk_start and end <= chunk_end:
                tags = [tag(n, N) for n in range(start, end + 1)]
                print(f"  Group {g+1:2d} [{start}-{end}]: ", end="")
                print(" ".join(tags))


# --------------------------------------
# Main
# --------------------------------------

if __name__ == "__main__":
    N = 2310  # change this to 30, 210, 2310, etc.

    view_blocks_of_complement(N)
    view_groups_of_6(N)
    view_groups_of_6_chunked_by_complement(N)
