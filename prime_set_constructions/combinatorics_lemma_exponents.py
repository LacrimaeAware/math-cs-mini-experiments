"""
combinatorics_lemma_exponents.py — "omission-block" groupings + factor exponents.

Extends combinatorics_lemma.py: besides the base grouping experiment, one term
of a grouping is made "active" and its small-prime factors are allowed exponents
1..max_exp_small. For every exponent pattern and every +/- sign pattern the
resulting N is tested for primality inside 1 < |N| < q^2, to see whether adding
exponents lets the construction "reach" more primes.
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from prime_lib import is_prime, next_prime

from itertools import product

# ---------- basic utilities ----------

def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result

# is_prime() and next_prime() are imported from prime_lib (see top of file).

# ---------- set partitions (for omission blocks) ----------

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

# ---------- rule: each element missing from exactly one group ----------

def valid_groupings(S):
    """
    Given a finite set S (e.g. {1,2,3,5,7}),
    return all valid groupings as tuples of frozensets, where each frozenset
    is the set of elements INCLUDED in that term.

    Rule:
      - Partition S into non-empty omission blocks M_i.
      - Each M_i is the set of elements MISSING from term i.
      - Each element of S is in exactly one M_i
        (=> missing from exactly one term).
      - Require at least 2 terms (>= 2 blocks).
      - Grouping is the collection of sets G_i = S \\ M_i.
    """
    S = set(S)
    groupings = set()

    for part in set_partitions(S):
        # need at least 2 groups (rule)
        if len(part) < 2:
            continue

        # part is a partition into omission-blocks M_i
        groups = [frozenset(S - block) for block in part]

        # canonicalize ordering: sort groups so same structure isn't repeated
        groups_sorted = tuple(sorted(groups, key=lambda x: sorted(x)))
        groupings.add(groups_sorted)

    # sort groupings: first by number of groups, then lexicographically
    return sorted(groupings, key=lambda gs: (len(gs), [sorted(x) for x in gs]))


# ---------- original no-exponent analysis (for comparison) ----------

def analyze_set(S):
    """
    Original lemma experiment:
      - build all valid groupings,
      - assign ± signs to term products (all exponents = 1),
      - count distinct |N| and primes below q^2.
    """
    S = set(S)
    print("Set S:", S)

    primes_in_S = [x for x in S if is_prime(x)]
    if not primes_in_S:
        raise ValueError("S must contain at least one prime > 1.")
    max_p = max(primes_in_S)
    q = next_prime(max_p)
    q2 = q * q

    print(f"Largest prime in S: {max_p}")
    print(f"Next prime q: {q}, q^2 = {q2}")
    print()

    groupings = valid_groupings(S)
    print(f"Number of valid groupings (structural / Bell-based): {len(groupings)}")

    all_values = []

    for g in groupings:
        term_values = [prod(term) for term in g]
        m = len(term_values)
        for signs in product([1, -1], repeat=m):
            val = sum(signs[i] * term_values[i] for i in range(m))
            all_values.append(val)

    print(f"Total values generated (with duplicates): {len(all_values)}")

    abs_values = [abs(v) for v in all_values]
    unique_abs = sorted(set(abs_values))
    print(f"Number of distinct absolute values: {len(unique_abs)}")

    nontrivial_abs = [x for x in unique_abs if x > 1]
    prime_abs = [x for x in nontrivial_abs if is_prime(x)]
    num_nontrivial = len(nontrivial_abs)
    num_prime_global = len(prime_abs)
    frac_prime_global = num_prime_global / num_nontrivial if num_nontrivial else 0.0

    print()
    print(f"Among all distinct |N| > 1:")
    print(f"  #nontrivial distinct |N|: {num_nontrivial}")
    print(f"  #primes among them:       {num_prime_global}")
    print(f"  fraction primes:          {frac_prime_global:.6f}")
    print()

    candidates = [v for v in all_values if 1 < abs(v) < q2]
    abs_candidates = sorted(set(abs(v) for v in candidates))

    print(f"Values with 1 < |N| < q^2 (including duplicates): {len(candidates)}")
    print(f"Distinct absolute values with 1 < |N| < q^2: {len(abs_candidates)}")

    primes_small = [x for x in abs_candidates if is_prime(x)]
    num_small = len(abs_candidates)
    num_prime_small = len(primes_small)
    frac_prime_small = num_prime_small / num_small if num_small else 0.0

    print("Distinct candidate |N| in (1, q^2) and primality:")
    for x in abs_candidates:
        print(f"  {x}: {'prime' if is_prime(x) else 'composite'}")
    print()
    print(f"Among distinct |N| with 1 < |N| < q^2:")
    print(f"  #candidates: {num_small}")
    print(f"  #primes:     {num_prime_small}")
    print(f"  fraction:    {frac_prime_small:.6f}")
    print()
    print("="*60)
    print()


# ---------- NEW: factor-level exponents on a single term ----------

def analyze_factor_exponents(
    S,
    small_primes=None,
    max_exp_small=2,
    max_terms=3,
    max_groupings=None,
    show_progress=True,
):
    """
    Exponent experiment:

    - groupings: same lemma rule (each element missing from exactly one term),
      but only groupings with number of terms <= max_terms.
    - pick ONE term in the grouping as "active".
      Only that term gets factor-level exponents.
    - For primes in 'small_primes' inside the active term, exponents run 1..max_exp_small.
      For all other primes in that term, exponent = 1.
      In all non-active terms, all exponents = 1.
    - For each exponent choice and each sign pattern over the terms, form

          N = sum_i (sign_i * term_i_value)

      and keep 1 < |N| < q^2.
    """

    S = set(S)
    primes_in_S = [x for x in S if is_prime(x)]
    if not primes_in_S:
        raise ValueError("S must contain at least one prime > 1.")
    max_p = max(primes_in_S)
    q = next_prime(max_p)
    q2 = q * q

    if small_primes is None:
        # default: allow small exponents only on the small primes in S
        small_primes = {p for p in S if p != 1 and p <= 7}
    else:
        small_primes = set(small_primes)

    print("Heuristic factor-exponent search for S =", S)
    print("  small_primes allowed exponents up to", max_exp_small, ":", small_primes)
    print("  max_terms per grouping:", max_terms)
    print("  q =", q, ", q^2 =", q2)
    print()

    # choose groupings
    all_groupings = valid_groupings(S)
    groupings = [g for g in all_groupings if len(g) <= max_terms]
    if max_groupings is not None:
        groupings = groupings[:max_groupings]

    print(f"  total groupings (all):     {len(all_groupings)}")
    print(f"  using groupings (filtered): {len(groupings)}")
    print()

    all_values = []
    prime_values = set()

    for gi, g in enumerate(groupings, start=1):
        m = len(g)
        term_primes = [sorted(term) for term in g]  # e.g. [[1,2,3,5], [1,2,5,7], ...]

        if show_progress and (gi == 1 or gi % 25 == 0 or gi == len(groupings)):
            print(f"  processing grouping {gi}/{len(groupings)} (m={m})")

        # choose one term to be "active" (gets factor exponents)
        for active_idx in range(m):
            # precompute base products for non-active terms
            base_term_values = []
            for ti, primes in enumerate(term_primes):
                if ti == active_idx:
                    base_term_values.append(None)  # placeholder
                else:
                    base_term_values.append(prod(primes))

            active_primes = term_primes[active_idx]

            # exponent options for each prime in the active term
            exp_options = []
            for p in active_primes:
                if p == 1:
                    # 1^anything = 1; exponent variation is meaningless
                    exp_options.append([1])
                elif p in small_primes:
                    exp_options.append(list(range(1, max_exp_small + 1)))
                else:
                    exp_options.append([1])

            # iterate over exponent patterns for the active term
            for exps in product(*exp_options):
                # build the actual term values
                term_values = list(base_term_values)
                active_val = 1
                for p, e in zip(active_primes, exps):
                    active_val *= p ** e
                term_values[active_idx] = active_val

                # now apply all sign patterns over the m terms
                for signs in product([1, -1], repeat=m):
                    N = sum(s * tv for s, tv in zip(signs, term_values))
                    if 1 < abs(N) < q2:
                        all_values.append(N)
                        if is_prime(abs(N)):
                            prime_values.add(abs(N))

    print()
    print("Total N with 1 < |N| < q^2 (with duplicates):", len(all_values))
    distinct_abs = sorted(set(abs(v) for v in all_values))
    print("Distinct |N| in (1, q^2):", len(distinct_abs))
    print()

    # primality stats in the window
    primes_in_window = [x for x in distinct_abs if is_prime(x)]
    num_candidates = len(distinct_abs)
    num_primes = len(primes_in_window)
    frac = num_primes / num_candidates if num_candidates else 0.0

    print("Distinct candidate |N| in (1, q^2) and primality:")
    for x in distinct_abs:
        print(f"  {x}: {'prime' if is_prime(x) else 'composite'}")
    print()
    print("Among distinct |N| with 1 < |N| < q^2 (factor-exponent search):")
    print(f"  #candidates: {num_candidates}")
    print(f"  #primes:     {num_primes}")
    print(f"  fraction:    {frac:.6f}")
    print()
    if primes_in_window:
        print("  primes (window):", primes_in_window)
    print()
    print("="*60)
    print()


if __name__ == "__main__":
    # Example: S up to 17
    S = {1, 2, 3, 5, 7, 11, 13, 17}

    print("=== BASE (no exponents) ===")
    analyze_set(S)

    print("=== FACTOR EXPONENTS (single active term, small primes only) ===")
    # You can tweak small_primes and max_exp_small.
    # For S up to 17 this should still be reasonably thorough.
    analyze_factor_exponents(
        S,
        small_primes={2, 3, 5, 7},
        max_exp_small=2,
        max_terms=3,
        max_groupings=None,   # set to an int if it gets slow
        show_progress=True,
    )
