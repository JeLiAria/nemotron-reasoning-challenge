"""eqsym_re_round2.py — round-2 reverse-engineering of the eq_symbol (symbol-cryptarithm) family.

Reproduces experiments #17-#20 of the write-up, offline, with the exact data:
  #17  global operator->operation map .......... 0/26 symbols   (operator is per-puzzle, not a global key)
  #18  classic digit-substitution decode ....... ~1/40          (base-10 cryptarithm model is wrong)
  #19  correct model under-determination ....... 340/823 ambiguous, 2/823 unique
  #20  learned rule-prior, held-out 50/50 ...... ~10.5%         (a prior recovers ~10%, vs 0% unique)

Model recovered (community structure-share): 5-char expressions `A(2) op B(2)` over a 26-symbol
alphabet; output = per-position char-op `out[i] = f(in[j], in[k])`, j,k in 0..4, and POSITION 2
(the operator) is an input VALUE to f, not a lookup key.

Run:  uv run python eqsym_re_round2.py
"""
import json, glob, sys, random
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from sym_core import CHAR_TO_INT, INT_TO_CHAR
from perpos_solver import BINARY_OPS

idx = CHAR_TO_INT
import os

PROB = os.environ.get("EQSYM_PROBLEMS_DIR", "problems")  # released: set via env


def load_cryptarithm():
    out = []
    for f in glob.glob(f"{PROB}/*.jsonl"):
        try: r = json.loads(open(f, encoding="utf-8").readline())
        except Exception: continue
        if not r.get("category", "").startswith("cryptarithm"): continue
        q = str(r["question"])
        if len(q) != 5: continue
        ex = [(str(e["input_value"]), str(e["output_value"])) for e in r["examples"]]
        out.append({"id": r["id"], "cat": r["category"], "ex": ex, "q": q,
                    "gold": str(r["answer"]).strip()})
    return out


def perpos_cands(exs, L):
    """Per output position, the (op,j,k) consistent with ALL given (expr,out) examples."""
    res = []
    for i in range(L):
        c = []
        for nm, fn in BINARY_OPS.items():
            for j in range(5):
                for k in range(5):
                    ok = True
                    for e, o in exs:
                        v = fn(idx[e[j]], idx[e[k]])
                        if v not in INT_TO_CHAR or INT_TO_CHAR[v] != o[i]: ok = False; break
                    if ok: c.append((nm, j, k))
        res.append(c)
    return res


# ---------------------------------------------------------------- #17
def probe_17_global_map(puz):
    """Does each operator symbol carry a consistent global per-position rule? (pool all 2-char outputs)"""
    trip = defaultdict(list)
    for p in puz:
        for e, o in p["ex"]:
            if len(o) == 2: trip[e[2]].append((e, o))
        if len(p["gold"]) == 2: trip[p["q"][2]].append((p["q"], p["gold"]))
    ok = tested = 0
    for sym, ts in trip.items():
        if len(ts) < 3: continue
        tested += 1
        # one global (op,j,k) per position consistent across all this symbol's triples?
        good = True
        for i in (0, 1):
            found = any(
                all(BINARY_OPS[op](idx[e[j]], idx[e[k]]) == idx[o[i]] for e, o in ts)
                for op in BINARY_OPS for j in range(5) for k in range(5))
            if not found: good = False; break
        ok += good
    return f"#17 global operator->op map: {ok}/{tested} symbols with a consistent global rule"


# ---------------------------------------------------------------- #18
def probe_18_digit_sub(puz, n=40, seed=1):
    """Classic base-10 digit-substitution (operator literal) + operand/result endian modes."""
    OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b}
    def num(digs, le):
        d = digs[::-1] if le else digs
        return int("".join(map(str, d))) if d else 0
    def fits(ad, bd, od, op):
        for ale in (0, 1):
            for ble in (0, 1):
                r = OPS[op](num(ad, ale), num(bd, ble))
                for ole in (0, 1):
                    if num(od, ole) == abs(r): return True
        return False
    pool = [p for p in puz if p["cat"] == "cryptarithm_deduce"]
    random.Random(seed).shuffle(pool)
    tested = solved = 0
    for p in pool:
        if tested >= n: break
        syms = set()
        for e, o in p["ex"]: syms |= set(e[:2] + e[3:5] + o)
        syms |= set(p["q"][:2] + p["q"][3:5])
        syms = sorted(syms)
        if len(syms) > 8: continue          # keep the brute-force cheap
        tested += 1
        for perm in permutations(range(10), len(syms)):
            m = dict(zip(syms, perm)); ok = True
            for e, o in p["ex"]:
                op = e[2]
                if op not in OPS or not fits([m[c] for c in e[:2]], [m[c] for c in e[3:5]],
                                             [m[c] for c in o], op): ok = False; break
            if ok:
                inv = {v: k for k, v in m.items()}; op = p["q"][2]
                if op in OPS:
                    r = abs(OPS[op](m[p["q"][0]] * 10 + m[p["q"][1]], m[p["q"][3]] * 10 + m[p["q"][4]]))
                    try:
                        if "".join(inv[int(c)] for c in str(r)) == p["gold"]: solved += 1
                    except Exception: pass
                break
    return f"#18 digit-substitution (+endian) on deduce<=8syms: {solved}/{tested} query-correct"


# ---------------------------------------------------------------- #19
def usable_deduce(puz):
    """Puzzles where same-op examples share one output length L == len(gold)."""
    out = []
    for p in puz:
        so = [(e, o) for e, o in p["ex"] if e[2] == p["q"][2]]
        if not so: continue
        Ls = {len(o) for e, o in so}
        if len(Ls) != 1: continue
        L = Ls.pop()
        if len(p["gold"]) != L: continue
        out.append((so, p["q"], p["gold"], L))
    return out


def probe_19_underdetermination(puz):
    """With the correct perpos model + same-op examples: unique vs ambiguous on the query."""
    out = Counter()
    for p in puz:
        so = [(e, o) for e, o in p["ex"] if e[2] == p["q"][2]]
        if not so: out["noex(guess)"] += 1; continue
        Ls = {len(o) for e, o in so}
        if len(Ls) != 1: out["varlen"] += 1; continue
        L = Ls.pop()
        cs = perpos_cands(so, L)
        if any(not c for c in cs): out["nofit"] += 1; continue
        determined = True
        for i in range(L):
            qp = {INT_TO_CHAR[BINARY_OPS[o](idx[p["q"][j]], idx[p["q"][k]])]
                  for o, j, k in cs[i] if BINARY_OPS[o](idx[p["q"][j]], idx[p["q"][k]]) in INT_TO_CHAR}
            if len(qp) != 1: determined = False; break
        out["unique" if determined else "ambiguous"] += 1
    return (f"#19 under-determination: ambiguous={out['ambiguous']}, unique={out['unique']}, "
            f"nofit={out['nofit']}, varlen={out['varlen']}, guess(no same-op)={out['noex(guess)']}")


# ---------------------------------------------------------------- #20
def probe_20_prior_heldout(puz, seed=7):
    """Build a frequency prior on a train half, predict the query on the test half (no answer peek)."""
    pz = usable_deduce(puz)
    random.Random(seed).shuffle(pz)
    h = len(pz) // 2; train, test = pz[:h], pz[h:]
    prior = Counter()
    for so, q, gold, L in train:
        cs = perpos_cands(so, L)
        for i in range(L):
            for (nm, j, k) in cs[i]:
                v = BINARY_OPS[nm](idx[q[j]], idx[q[k]])
                if v in INT_TO_CHAR and INT_TO_CHAR[v] == gold[i]: prior[(nm, j, k)] += 1
    rank = {t: r for r, (t, _) in enumerate(prior.most_common())}
    def ev(S):
        cor = 0
        for so, q, gold, L in S:
            cs = perpos_cands(so, L); pred = []; ok = True
            for i in range(L):
                if not cs[i]: ok = False; break
                best = min(cs[i], key=lambda t: rank.get(t, 1 << 30))
                v = BINARY_OPS[best[0]](idx[q[best[1]]], idx[q[best[2]]])
                pred.append(INT_TO_CHAR.get(v, "?"))
            cor += ok and "".join(pred) == gold
        return cor
    ctr, cte = ev(train), ev(test)
    return (f"#20 learned prior held-out: train={ctr}/{len(train)}={100*ctr/max(1,len(train)):.1f}%  "
            f"TEST={cte}/{len(test)}={100*cte/max(1,len(test)):.1f}%  (usable deduce={len(pz)})")


if __name__ == "__main__":
    puz = load_cryptarithm()
    print(f"loaded {len(puz)} cryptarithm puzzles "
          f"(deduce={sum(p['cat']=='cryptarithm_deduce' for p in puz)}, "
          f"guess={sum(p['cat']=='cryptarithm_guess' for p in puz)})\n")
    print(probe_17_global_map(puz))
    print(probe_18_digit_sub(puz))
    print(probe_19_underdetermination(puz))
    print(probe_20_prior_heldout(puz))
