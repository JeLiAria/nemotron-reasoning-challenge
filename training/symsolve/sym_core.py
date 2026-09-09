"""Core helpers for symbol puzzles: char↔digit mapping, base-26 conversion."""

# Alphabet mapping (from file header)
ALPHA = "!\"#$%&'()*+-/:<>?@[\\]^`{|}"
assert len(ALPHA) == 26, f"alphabet has {len(ALPHA)} chars: {ALPHA!r}"
CHAR_TO_INT = {c: i for i, c in enumerate(ALPHA)}
INT_TO_CHAR = {i: c for c, i in CHAR_TO_INT.items()}


def chars_to_int(s):
    """Convert string of base-26 chars to integer (big-endian)."""
    n = 0
    for c in s:
        n = n * 26 + CHAR_TO_INT[c]
    return n


def int_to_chars(n, length=None):
    """Convert integer to string of base-26 chars. If length given, pad with leading 0-char."""
    if n < 0:
        raise ValueError(f"cannot encode negative {n}")
    if n == 0:
        s = INT_TO_CHAR[0]
    else:
        s = ""
        while n > 0:
            s = INT_TO_CHAR[n % 26] + s
            n //= 26
    if length is not None:
        s = INT_TO_CHAR[0] * (length - len(s)) + s
    return s


def split_expr(expr):
    """Split a 5-char expression into (a_str, op_str, b_str)."""
    if len(expr) != 5:
        raise ValueError(f"expr must be 5 chars: {expr!r}")
    return expr[:2], expr[2], expr[3:]


if __name__ == "__main__":
    # Sanity checks
    print(f"Alphabet ({len(ALPHA)} chars): {ALPHA!r}")
    for c, v in [("!", 0), ("}", 25), ("`", 22), ("\\", 19), ("'", 6)]:
        assert CHAR_TO_INT[c] == v, f"{c!r} → {CHAR_TO_INT[c]} (expected {v})"
    print("All char-to-int mappings OK")

    # Test base-26 conversion
    print(f"chars_to_int('!{INT_TO_CHAR[1]}') = {chars_to_int('!' + INT_TO_CHAR[1])}")  # 0*26+1=1
    print(f"chars_to_int('}}') = {chars_to_int('}}')}")  # 25
    print(f"chars_to_int('}}}}') = {chars_to_int('}}}}')}")  # 25*26+25 = 675
    print(f"int_to_chars(675) = {int_to_chars(675)!r}")
    print(f"int_to_chars(0) = {int_to_chars(0)!r}")

    # Try one example: \'-!`
    a, op, b = split_expr("\\'-!`")
    print(f"\\'-!`: a={a!r} ({chars_to_int(a)})  op={op!r}  b={b!r} ({chars_to_int(b)})")
    # Output should be \\
    print(f"Expected output \\\\ = {chars_to_int(chr(0x5c)*2)}")
