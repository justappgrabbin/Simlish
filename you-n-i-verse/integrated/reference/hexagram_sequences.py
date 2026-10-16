"""
hexagram_sequences.py — Fu Xi, King Wen, and Mawangdui orderings.

King Wen binary values are VERIFIED, pulled directly from a maintained
public dataset (adamblvck/iching-wilhelm-dataset, MIT licensed), not
hand-transcribed from images. Fu Xi and Mawangdui are then GENERATED
from verified structural principles rather than separately transcribed —
consistent with this system's "compute, don't hardcode" rule.

Binary convention (confirmed against source): leftmost char = top line
(line 6), rightmost = bottom line (line 1). Hexagram binary = upper
trigram bits (3) + lower trigram bits (3).
"""

# ── VERIFIED KING WEN DATA (source: adamblvck/iching-wilhelm-dataset) ──────
# hex number -> 6-bit binary string
KING_WEN_BINARY = {
    1: "111111", 2: "000000", 3: "010001", 4: "100010", 5: "010111",
    6: "111010", 7: "000010", 8: "010000", 9: "110111", 10: "111011",
    11: "000111", 12: "111000", 13: "111101", 14: "101111", 15: "000100",
    16: "001000", 17: "011001", 18: "100110", 19: "000011", 20: "110000",
    21: "101001", 22: "100101", 23: "100000", 24: "000001", 25: "111001",
    26: "100111", 27: "100001", 28: "011110", 29: "010010", 30: "101101",
    31: "011100", 32: "001110", 33: "111100", 34: "001111", 35: "101000",
    36: "000101", 37: "110101", 38: "101011", 39: "010100", 40: "001010",
    41: "100011", 42: "110001", 43: "011111", 44: "111110", 45: "011000",
    46: "000110", 47: "011010", 48: "010110", 49: "011101", 50: "101110",
    51: "001001", 52: "100100", 53: "110100", 54: "001011", 55: "001101",
    56: "101100", 57: "110110", 58: "011011", 59: "110010", 60: "010011",
    61: "110011", 62: "001100", 63: "010101", 64: "101010",
}

BINARY_TO_KING_WEN = {v: k for k, v in KING_WEN_BINARY.items()}

# ── TRIGRAM DEFINITIONS (3-bit, leftmost = top line) ────────────────────────
TRIGRAMS = {
    "Qian": "111",  # Heaven
    "Gen":  "100",  # Mountain
    "Kan":  "010",  # Water
    "Zhen": "001",  # Thunder
    "Kun":  "000",  # Earth
    "Dui":  "011",  # Lake
    "Li":   "101",  # Fire
    "Xun":  "110",  # Wind
}


def fu_xi_order() -> list:
    """
    GENERATED, not transcribed: pure binary counting, 0 to 63.
    Verified against source: binary 0 (Kun) comes first, binary 63
    (Qian, King Wen #1) comes last.
    """
    return [BINARY_TO_KING_WEN[format(n, "06b")] for n in range(64)]


def king_wen_order() -> list:
    """The reference order itself — 1 through 64, no transformation needed."""
    return list(range(1, 65))


def mawangdui_order() -> list:
    """
    GENERATED from the verified structural principle (not a hardcoded
    table): grouped by UPPER trigram in family order Qian, Gen, Kan,
    Zhen, Kun, Dui, Li, Xun. Within each octet, the LOWER trigram cycles
    through the SAME family order, with the pure doubled trigram first.
    """
    family_order = ["Qian", "Gen", "Kan", "Zhen", "Kun", "Dui", "Li", "Xun"]
    sequence = []

    for upper_name in family_order:
        upper_bits = TRIGRAMS[upper_name]
        # Pure doubled trigram first, then the rest of family order
        lower_sequence = [upper_name] + [t for t in family_order if t != upper_name]

        for lower_name in lower_sequence:
            lower_bits = TRIGRAMS[lower_name]
            full_binary = upper_bits + lower_bits
            king_wen_num = BINARY_TO_KING_WEN[full_binary]
            sequence.append(king_wen_num)

    return sequence


if __name__ == "__main__":
    fx = fu_xi_order()
    kw = king_wen_order()
    mw = mawangdui_order()

    print(f"Fu Xi order (first 8, by King Wen number): {fx[:8]}")
    print(f"  Sanity check: binary 0 -> King Wen #{fx[0]} (should be 2, Kun)")
    print(f"  Sanity check: binary 63 -> King Wen #{fx[-1]} (should be 1, Qian)")
    print()

    print(f"King Wen order (first 8): {kw[:8]}")
    print()

    print(f"Mawangdui order (first 8, by King Wen number): {mw[:8]}")
    print(f"  First octet should start with pure Qian (King Wen #1)")
    print(f"  Full first octet (Qian family): {mw[:8]}")
    print()

    print(f"All three sequences have 64 unique entries: "
          f"FuXi={len(set(fx))==64}, KingWen={len(set(kw))==64}, Mawangdui={len(set(mw))==64}")
