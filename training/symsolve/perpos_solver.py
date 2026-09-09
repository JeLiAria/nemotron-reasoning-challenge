"""Perpos_v2 solver: for each output character position, find a binary op on input char positions."""
from sym_core import CHAR_TO_INT, INT_TO_CHAR, split_expr


# Binary ops on (0-25, 0-25) → 0-25
BINARY_OPS = {
    "max":          lambda a, b: max(a, b),
    "min":          lambda a, b: min(a, b),
    "add_mod26":    lambda a, b: (a + b) % 26,
    "sub_AB_mod26": lambda a, b: (a - b) % 26,
    "sub_BA_mod26": lambda a, b: (b - a) % 26,
    "abs_diff":     lambda a, b: abs(a - b),
    "mul_mod26":    lambda a, b: (a * b) % 26,
    "xor_5bit":     lambda a, b: a ^ b,
    "and_5bit":     lambda a, b: a & b,
    "or_5bit":      lambda a, b: a | b,
    "first":        lambda a, b: a,
    "second":       lambda a, b: b,
}

# Unary ops (mostly subsumed by binary with same args)
UNARY_OPS = {
    "id":     lambda a: a,
    "c25":    lambda a: 25 - a,
    "neg_mod26": lambda a: (-a) % 26,
}


def find_perpos_rules(examples, query=None, expected=None):
    """For each example, expr is 5-char string, output is variable-length string.

    Try to fit a rule per output position. Filters examples by query operator (in[2]).
    """
    if not examples:
        return None
    # Filter by query operator if provided
    if query is not None:
        query_op = query[2]
        examples = [ex for ex in examples if ex["expr"][2] == query_op]
    # Convert examples to int arrays
    parsed_ex = []
    for ex in examples:
        in_chars = [CHAR_TO_INT[c] for c in ex["expr"]]
        out_chars = [CHAR_TO_INT[c] for c in ex["output"]]
        parsed_ex.append((in_chars, out_chars))

    # If we have a query+expected, include them as a constraint too
    extra = []
    if query is not None and expected is not None:
        in_chars_q = [CHAR_TO_INT[c] for c in query]
        out_chars_q = [CHAR_TO_INT[c] for c in expected]
        extra.append((in_chars_q, out_chars_q))
    # All constraints (examples + query if available)
    all_constraints = parsed_ex + extra

    if not all_constraints:
        return None
    # Group by output length
    out_lens = set(len(o) for _, o in all_constraints)
    if len(out_lens) > 1:
        return None  # variable length even after filtering, skip
    L = out_lens.pop()

    # For each output position i (0..L-1), find an (op, j, k) consistent across all_constraints
    rules = []
    for i in range(L):
        found = None
        for op_name, op_fn in BINARY_OPS.items():
            for j in range(5):
                for k in range(5):
                    if all(op_fn(in_chars[j], in_chars[k]) == out_chars[i]
                           for in_chars, out_chars in all_constraints):
                        found = (op_name, j, k)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            return None
        rules.append(found)
    return {"length": L, "rules": rules, "matches_query": (extra != [])}


def apply_perpos(rules, query):
    in_chars = [CHAR_TO_INT[c] for c in query]
    return "".join(INT_TO_CHAR[BINARY_OPS[op_name](in_chars[j], in_chars[k])]
                   for op_name, j, k in rules)


if __name__ == "__main__":
    from parse_sym import parse_puzzles
    pz = parse_puzzles()
    # Test on a known SOLVED perpos puzzle
    # 6d6d0531: rule=out[0]=max(in[2],in[2]) | out[1]=sub_AB_mod26(in[3],in[4]) | out[2]=max(in[4],in[0])
    target = next(p for p in pz if p["id"] == "6d6d0531")
    result = find_perpos_rules(target["examples"], target["query"], target["expected"])
    print(f"6d6d0531: {result}")
    if result and result["rules"]:
        for i, r in enumerate(result["rules"]):
            print(f"  out[{i}] = {r[0]}(in[{r[1]}], in[{r[2]}])")
        pred = apply_perpos(result["rules"], target["query"])
        print(f"  Query pred: {pred!r}  expected: {target['expected']!r}")
