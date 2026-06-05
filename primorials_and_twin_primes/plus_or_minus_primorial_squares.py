"""
plus_or_minus_primorial_squares.py — twin-prime factors of p_n# +/- small offsets.

For each n in [MIN_N, MAX_N], take the n-th primorial p_n# and form N = p_n# +/-
offset for each configured offset. Factor N (sympy, with a per-case timeout and
an on-disk cache) and check whether any prime factor r belongs to a twin-prime
pair (r-2 or r+2 prime), excluding a configurable small-prime set. Reports which
primorials never produced a twin-prime factor. Caches and a live log are written
to outputs/primorial_squares_cache/.
"""

# Project-local shared output directory (see prime_lib.py at repo root).
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from prime_lib import ensure_output_dir

import os, json, time, math, hashlib, multiprocessing as mp
from sympy import prime, primorial, factorint, isprime, primerange
from sympy.ntheory import pollard_rho

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# =========================
# YOU EDIT THIS SECTION
# =========================

MIN_N = 1
MAX_N = 50

OFFSETS = [5, 6]          # check all of these offsets
DO_PLUS = True
DO_MINUS = True

# Exclude these primes from counting as "twin prime factors"
EXCLUDE_PRIMES = [2, 3, 5]
EXCLUDE_PN = True         # also exclude current p_n itself from counting

FULL_FACTOR = False
QUICK_TRIAL_LIMIT = 10000
TIMEOUT_SEC = 100

PRINT_EVERY = 1
SAVE_EVERY = 10

CACHE_DIR = str(ensure_output_dir("primorial_squares_cache"))

# =========================


def ensure_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def cfg_signature():
    cfg = {
        "MIN_N": MIN_N, "MAX_N": MAX_N,
        "OFFSETS": OFFSETS, "DO_PLUS": DO_PLUS, "DO_MINUS": DO_MINUS,
        "EXCLUDE_PRIMES": EXCLUDE_PRIMES, "EXCLUDE_PN": EXCLUDE_PN,
        "FULL_FACTOR": FULL_FACTOR, "QUICK_TRIAL_LIMIT": QUICK_TRIAL_LIMIT,
        "TIMEOUT_SEC": TIMEOUT_SEC
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


SIG = f"indexprim_{cfg_signature()}"
PRIM_CACHE_FILE = os.path.join(CACHE_DIR, f"primorial_cache_{SIG}.json")
RES_CACHE_FILE  = os.path.join(CACHE_DIR, f"results_cache_{SIG}.json")
LIVE_LOG_FILE   = os.path.join(CACHE_DIR, f"live_log_{SIG}.txt")


def log_line(s):
    print(s, flush=True)
    with open(LIVE_LOG_FILE, "a") as f:
        f.write(s + "\n")


def primorial_index(n: int) -> int:
    return int(primorial(n, nth=True))


def nth_prime(n: int) -> int:
    return int(prime(n))


def sanity_check_index_primorial():
    known = {1: 2, 2: 6, 3: 30, 4: 210, 5: 2310}
    for n, v in known.items():
        got = primorial_index(n)
        if got != v:
            raise RuntimeError(
                f"[SANITY FAIL] primorial_index({n})={got}, expected {v}"
            )


def num_digits(x: int) -> int:
    x = abs(x)
    if x == 0:
        return 1
    return int(math.log10(x)) + 1


def has_twin_prime_factor(factors, exclude_set):
    for r in factors.keys():
        if r in exclude_set:
            continue
        if isprime(r - 2) or isprime(r + 2):
            return True
    return False


def quick_factor(n, trial_limit=10000):
    facs = {}
    x = n

    for p in primerange(2, trial_limit + 1):
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            facs[p] = e
        if x == 1:
            return facs, True

    if x != 1:
        if isprime(x):
            facs[x] = facs.get(x, 0) + 1
            return facs, True

        d = pollard_rho(x)
        if d is None or d == x:
            facs[x] = facs.get(x, 0) + 1
            return facs, False

        y = x // d
        facs[d] = facs.get(d, 0) + 1
        facs[y] = facs.get(y, 0) + 1
        return facs, False

    return facs, True


def _factor_worker(n, full_factor, trial_limit, out_q):
    try:
        if full_factor:
            facs = factorint(n)
            out_q.put((facs, True))
        else:
            facs, fully = quick_factor(n, trial_limit=trial_limit)
            out_q.put((facs, fully))
    except Exception:
        # always put something so parent never blocks
        out_q.put(({}, False))


def factor_with_timeout(n, full_factor, trial_limit, timeout_sec=10):
    out_q = mp.Queue()
    proc = mp.Process(target=_factor_worker, args=(n, full_factor, trial_limit, out_q))
    proc.daemon = True
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {}, False, True

    try:
        facs, fully = out_q.get(timeout=1.0)
    except Exception:
        return {}, False, True

    return facs, fully, False


def fmt_factors(facs, limit_terms=8):
    items = sorted(facs.items(), key=lambda kv: kv[0])
    parts = []
    for i, (k, v) in enumerate(items):
        if i >= limit_terms:
            parts.append("...")
            break
        parts.append(f"{k}^{v}" if v != 1 else str(k))
    return " * ".join(parts) if parts else "1"


def run():
    ensure_dir()
    sanity_check_index_primorial()

    prim_cache = load_json(PRIM_CACHE_FILE, {})
    res_cache  = load_json(RES_CACHE_FILE, {})

    with open(LIVE_LOG_FILE, "w") as f:
        f.write(f"=== live run log (index primorial, sig={SIG}) ===\n")

    n_list = list(range(MIN_N, MAX_N + 1))

    tasks = []
    for n in n_list:
        p_n = nth_prime(n)
        for off in OFFSETS:
            if DO_PLUS:
                tasks.append((n, p_n, +1, off))
            if DO_MINUS:
                tasks.append((n, p_n, -1, off))

    iterator = tasks
    if tqdm is not None:
        iterator = tqdm(tasks, desc="p_n# ± offsets", unit="case")

    computed_since_save = 0
    per_n_success = {n: False for n in n_list}
    t0 = time.time()

    for idx, (n, p_n, sign, off) in enumerate(iterator, start=1):
        nkey = str(n)
        if nkey in prim_cache:
            P = int(prim_cache[nkey])
        else:
            P = primorial_index(n)
            prim_cache[nkey] = P

        N = P + sign * off
        digits = num_digits(N)
        cache_key = f"n={n}|p_n={p_n}|{'+' if sign>0 else '-'}|off={off}"

        # cached?
        if cache_key in res_cache:
            entry = res_cache[cache_key]
            twin_present = entry["has_twin_prime_factor_excl3"]
            if twin_present:
                per_n_success[n] = True

            if idx % PRINT_EVERY == 0:
                facs = {int(k): v for k, v in entry["factors"].items()}
                fac_str = fmt_factors(facs)
                flag = "twin" if twin_present else "NO-twin"
                log_line(
                    f"[cached] n={n} p_n={p_n}  p_n# {'+' if sign>0 else '-'} {off} "
                    f"({digits} digits) factors: {fac_str} -> {flag}"
                )
            continue

        # trivial N cases (0, ±1) — skip factoring
        if abs(N) <= 1:
            entry = {
                "n": n, "p_n": p_n, "primorial": P,
                "offset": off, "sign": "+" if sign>0 else "-",
                "N": int(N),
                "factors": {},
                "fully_factored": True,
                "timed_out": False,
                "has_twin_prime_factor_excl3": False
            }
            res_cache[cache_key] = entry
            computed_since_save += 1
            if idx % PRINT_EVERY == 0:
                log_line(
                    f"[{idx}/{len(tasks)}] n={n} p_n={p_n}  p_n# {'+' if sign>0 else '-'} {off} "
                    f"({digits} digits) N={N} -> trivial skip"
                )
            continue

        log_line(
            f"-> factoring n={n} p_n={p_n} p_n# {'+' if sign>0 else '-'} {off} ({digits} digits) ..."
        )

        facs, fully, timed_out = factor_with_timeout(
            N,
            full_factor=FULL_FACTOR,
            trial_limit=QUICK_TRIAL_LIMIT,
            timeout_sec=TIMEOUT_SEC
        )

        if timed_out:
            entry = {
                "n": n, "p_n": p_n, "primorial": P,
                "offset": off, "sign": "+" if sign>0 else "-",
                "N": int(N),
                "factors": {},
                "fully_factored": False,
                "timed_out": True,
                "has_twin_prime_factor_excl3": None
            }
            res_cache[cache_key] = entry
            computed_since_save += 1
            log_line(f"[{idx}/{len(tasks)}] TIMED OUT after {TIMEOUT_SEC}s; skipping\n")
            continue

        exclude_set = set(EXCLUDE_PRIMES)
        if EXCLUDE_PN:
            exclude_set.add(p_n)

        twin_present = has_twin_prime_factor(facs, exclude_set)
        if twin_present:
            per_n_success[n] = True

        entry = {
            "n": n, "p_n": p_n, "primorial": P,
            "offset": off, "sign": "+" if sign>0 else "-",
            "N": int(N),
            "factors": {str(k): int(v) for k, v in facs.items()},
            "fully_factored": bool(fully),
            "timed_out": False,
            "has_twin_prime_factor_excl3": bool(twin_present)
        }
        res_cache[cache_key] = entry
        computed_since_save += 1

        if idx % PRINT_EVERY == 0:
            fac_str = fmt_factors(facs)
            flag = "twin" if twin_present else "NO-twin"
            full_tag = "FULL" if fully else "PARTIAL"
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed else 0.0
            log_line(
                f"[{idx}/{len(tasks)}] n={n} p_n={p_n} ({digits} digits) {full_tag} "
                f"factors: {fac_str} -> {flag} | rate {rate:.2f} cases/s\n"
            )

        if computed_since_save >= SAVE_EVERY:
            save_json(PRIM_CACHE_FILE, prim_cache)
            save_json(RES_CACHE_FILE, res_cache)
            computed_since_save = 0
            log_line(f"--- saved caches at case {idx} ---\n")

    save_json(PRIM_CACHE_FILE, prim_cache)
    save_json(RES_CACHE_FILE, res_cache)

    log_line("=== run complete; caches saved ===")

    no_hit = []
    for n in n_list:
        if not per_n_success[n]:
            no_hit.append((n, nth_prime(n)))

    if no_hit:
        log_line("\nPrimorials where NONE of the enabled (±, offsets) had a twin-prime factor:", )
        for n, p_n in no_hit:
            log_line(f"  n={n}, p_n={p_n}")
    else:
        log_line("\nEvery primorial in range had at least one twin-prime-factor hit.")


if __name__ == "__main__":
    run()
