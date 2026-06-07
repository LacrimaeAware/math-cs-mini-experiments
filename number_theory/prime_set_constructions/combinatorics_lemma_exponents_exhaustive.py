"""
combinatorics_lemma_exponents_exhaustive.py — exhaustive factor-exponent search.

The most thorough member of the "lemma" family: over all groupings with up to
MAX_TERMS_EXP terms, it sweeps factor exponents on every term's small primes and
all +/- sign patterns, recording up to a few human-readable representations of
each prime it reaches in the window 1 < |N| < q^2. Config knobs are at the top.
"""

import sys
import pathlib
sys.path.append(next(str(p) for p in pathlib.Path(__file__).resolve().parents if (p / "prime_lib.py").exists()))
from prime_lib import is_prime, next_prime

from itertools import product

# ================== CONFIG ==================

S = {1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

# primes allowed to have exponents > 1
SMALL_PRIMES = (1, 2, 3, 5, 7, 11, 13, 17)

# max exponent for those small primes
MAX_EXP_SMALL = 3

# max number of terms in a grouping we will process with exponents
MAX_TERMS_EXP = 5

# how often to print progress (in number of groupings)
PROGRESS_EVERY = 1000

# max stored representations per prime (to avoid huge output)
MAX_REPS_PER_PRIME = 2


# ================== BASIC UTILITIES ==================

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

# is_prime() and next_prime() are imported from prime_lib (see top of file).


# ================== SET PARTITIONS / GROUPINGS ==================

def set_partitions(collection):
    """
    Generate all set partitions of 'collection'.
    Each partition is a list of frozensets whose union is collection
    and which are pairwise disjoint.
    """
    collection = list(collection)
    if not collection:
        yield []
        return

    first = collection[0]
    for rest in set_partitions(collection[1:]):
        # put 'first' into each existing block
        for i in range(len(rest)):
            new_part = list(rest)
            new_block = set(new_part[i])
            new_block.add(first)
            new_part[i] = frozenset(new_block)
            yield new_part
        # or start a new block with 'first' alone
        yield [frozenset({first}), *rest]


def valid_groupings(S):
    """
    Rule:

      - Partition S into omission blocks M_i.
      - Each M_i is the set of elements missing from term i.
      - Each element of S is in exactly one M_i.
      - At least 2 terms.
      - Terms G_i = S \\ M_i.

    Returns: list of groupings, each grouping is a tuple of frozensets.
    """
    S = set(S)
    groupings = set()

    for part in set_partitions(S):
        if len(part) < 2:
            continue

        groups = [frozenset(S - block) for block in part]
        groups_sorted = tuple(sorted(groups, key=lambda x: sorted(x)))
        groupings.add(groups_sorted)

    return sorted(groupings, key=lambda gs: (len(gs), [sorted(x) for x in gs]))


# ================== EXPONENT SEARCH W/ REPRESENTATIONS ==================

def expression_string(grouping, active_idx, exp_map_active, signs):
    """
    Build a human-readable expression string:
      ±(prod of primes^exponents) ± ...
    """
    terms_str = []
    for j, term in enumerate(grouping):
        factors = []
        for p in sorted(term):
            if j == active_idx and p in exp_map_active:
                e = exp_map_active[p]
            else:
                e = 1
            if e == 1:
                factors.append(str(p))
            else:
                factors.append(f"{p}^{e}")
        term_str = "*".join(factors) if factors else "1"

        sign = signs[j]
        if j == 0:
            # first term: use "+" or "-" prefix explicitly
            if sign == 1:
                terms_str.append(term_str)
            else:
                terms_str.append(f"-{term_str}")
        else:
            if sign == 1:
                terms_str.append(f"+{term_str}")
            else:
                terms_str.append(f"-{term_str}")

    return " ".join(terms_str)


def factor_exponent_search_with_reps(
    S, small_primes, max_exp_small, max_terms_exp,
    progress_every=1000, max_reps_per_prime=3
):
    S = set(S)
    primes_in_S = [x for x in S if is_prime(x)]
    if not primes_in_S:
        raise ValueError("S must contain at least one prime > 1.")
    max_p = max(primes_in_S)
    q = next_prime(max_p)
    q2 = q * q

    print("=== EXHAUSTIVE-ish FACTOR EXPONENT SEARCH WITH REPS ===")
    print("S:", S)
    print("largest prime in S:", max_p)
    print("q (next prime):", q, ",  q^2 =", q2)
    print("small_primes with exponents:", set(small_primes))
    print("max_exp_small:", max_exp_small)
    print("max_terms_exp:", max_terms_exp)
    print()

    all_groupings = valid_groupings(S)
    total_groupings = len(all_groupings)
    groupings = [g for g in all_groupings if len(g) <= max_terms_exp]
    used_groupings = len(groupings)

    print("total groupings (all):     ", total_groupings)
    print("using groupings (filtered):", used_groupings)
    print()

    total_N_in_window = 0
    candidates = set()
    prime_to_reps = {}  # prime -> list of expression strings

    for idx, grouping in enumerate(groupings, start=1):
        m = len(grouping)

        # precompute small primes per term
        term_small_factors = [
            sorted(set(term) & set(small_primes))
            for term in grouping
        ]

        # for each term as "active" exponent term
        for active_idx in range(m):
            active_small = term_small_factors[active_idx]
            if not active_small:
                continue

            # exponent choices for active term small primes
            exp_choices = {
                p: list(range(1, max_exp_small + 1))
                for p in active_small
            }
            keys = list(exp_choices.keys())
            choice_lists = [exp_choices[k] for k in keys]

            for exp_tuple in product(*choice_lists):
                exp_map_active = dict(zip(keys, exp_tuple))

                # numeric term values
                term_values = []
                for j, term in enumerate(grouping):
                    val = 1
                    for p in term:
                        if j == active_idx and p in exp_map_active:
                            e = exp_map_active[p]
                        else:
                            e = 1
                        val *= (p ** e)
                    term_values.append(val)

                # sign patterns
                for signs in product((1, -1), repeat=m):
                    N = 0
                    for j in range(m):
                        N += signs[j] * term_values[j]

                    absN = abs(N)
                    if 1 < absN < q2:
                        total_N_in_window += 1
                        candidates.add(absN)
                        if is_prime(absN):
                            if absN not in prime_to_reps:
                                prime_to_reps[absN] = []
                            if len(prime_to_reps[absN]) < max_reps_per_prime:
                                expr = expression_string(
                                    grouping, active_idx, exp_map_active, signs
                                )
                                prime_to_reps[absN].append(expr)

        # progress
        if idx % progress_every == 0 or idx == used_groupings:
            pct = 100.0 * idx / used_groupings
            print(f"Processed groupings: {idx}/{used_groupings} ({pct:.1f}%)")

    print()
    print("Total N with 1 < |N| < q^2 (with duplicates):", total_N_in_window)
    print("Distinct |N| in (1, q^2):", len(candidates))
    print()

    primes_in_window = sorted(p for p in candidates if is_prime(p))
    print("Primes in (1, q^2) hit by this search (sorted):")
    print(primes_in_window)
    print()

    print("Representations (up to", max_reps_per_prime, "per prime):")
    for p in primes_in_window:
        reps = prime_to_reps.get(p, [])
        print(f"\nPrime {p}:")
        if not reps:
            print("  (no stored reps — hit only via non-logged configs)")
        else:
            for r in reps:
                print("  ", r)


if __name__ == "__main__":
    factor_exponent_search_with_reps(
        S=S,
        small_primes=SMALL_PRIMES,
        max_exp_small=MAX_EXP_SMALL,
        max_terms_exp=MAX_TERMS_EXP,
        progress_every=PROGRESS_EVERY,
        max_reps_per_prime=MAX_REPS_PER_PRIME,
    )
