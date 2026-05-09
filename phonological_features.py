"""
phonological_features.py
========================
Defines the 35 phonological features from Table 1 of Shahin et al. (2025)
and provides the phoneme-to-feature mapping for the 39-phoneme CMU set.

Feature categories (paper Table 1):
  Manners: consonant, sonorant, fricative, nasal, stop, approximant,
           affricate, liquid, vowel, semivowel, continuant
  Places:  alveolar, palatal, dental, glottal, labial, velar, mid, high,
           low, front, back, central, anterior, posterior, retroflex,
           bilabial, coronal, dorsal
  Others:  long, short, monophthong, diphthong, round, voiced

The model output has 71 nodes: 35 (+att) + 35 (-att) + 1 (shared blank).
"""

# ─────────────────────────────────────────────────────────────────────────────
# The 35 phonological features (paper Table 1), in a fixed canonical order
# ─────────────────────────────────────────────────────────────────────────────
PHONOLOGICAL_FEATURES = [
    # Manners (11)
    "consonant", "sonorant", "fricative", "nasal", "stop",
    "approximant", "affricate", "liquid", "vowel", "semivowel", "continuant",
    # Places (18)
    "alveolar", "palatal", "dental", "glottal", "labial", "velar",
    "mid", "high", "low", "front", "back", "central",
    "anterior", "posterior", "retroflex", "bilabial", "coronal", "dorsal",
    # Others (6)
    "long", "short", "monophthong", "diphthong", "round", "voiced",
]
assert len(PHONOLOGICAL_FEATURES) == 35, "Must have exactly 35 features"

FEATURE_TO_IDX = {feat: i for i, feat in enumerate(PHONOLOGICAL_FEATURES)}
NUM_FEATURES = len(PHONOLOGICAL_FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
# Output node layout (paper Section 3.3):
#   nodes 0..34        → +att for features 0..34
#   nodes 35..69       → -att for features 0..34
#   node  70           → shared blank
# ─────────────────────────────────────────────────────────────────────────────
NUM_OUTPUT_NODES = 71   # 35 + 35 + 1
BLANK_IDX = 70

def feature_idx_to_pos_node(feat_idx: int) -> int:
    """Return output node index for +att of a given feature."""
    return feat_idx

def feature_idx_to_neg_node(feat_idx: int) -> int:
    """Return output node index for -att of a given feature."""
    return feat_idx + NUM_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# CMU 39-phoneme set  (TIMIT 61→39 reduced set used in the paper)
# ─────────────────────────────────────────────────────────────────────────────
CMU_39_PHONEMES = [
    "aa", "ae", "ah", "aw", "ay","ao",
    "b",  "ch", "d",  "dh", "eh",
    "er", "ey", "f",  "g",  "hh",
    "ih", "iy", "jh", "k",  "l",
    "m",  "n",  "ng", "ow", "oy",
    "p",  "r",  "s",  "sh", "t",
    "th", "uh", "uw", "v",  "w",
    "y",  "z",  "zh",
]
PHONEME_TO_IDX = {p: i for i, p in enumerate(CMU_39_PHONEMES)}
NUM_PHONEMES = len(CMU_39_PHONEMES)   # 39


# ─────────────────────────────────────────────────────────────────────────────
# Phoneme → phonological feature binary vector
# Each phoneme maps to a dict {feature_name: True/False}.
# Derived from standard phonological feature charts (Chomsky & Halle 1968,
# as referenced in the paper).
# ─────────────────────────────────────────────────────────────────────────────
def _p(features_present: list[str]) -> dict[str, bool]:
    """Helper: build feature dict from list of present features."""
    return {f: (f in features_present) for f in PHONOLOGICAL_FEATURES}


PHONEME_FEATURES: dict[str, dict[str, bool]] = {
    # ── Stops ──────────────────────────────────────────────────────────────
    "p":  _p(["consonant", "stop", "labial", "anterior", "bilabial"]),
    "b":  _p(["consonant", "stop", "labial", "anterior", "bilabial",
               "voiced"]),
    "t":  _p(["consonant", "stop", "alveolar", "anterior", "coronal"]),
    "d":  _p(["consonant", "stop", "alveolar", "anterior", "coronal",
               "voiced"]),
    "k":  _p(["consonant", "stop", "velar", "posterior", "dorsal"]),
    "g":  _p(["consonant", "stop", "velar", "posterior", "dorsal",
               "voiced"]),
 
    # ── Fricatives ─────────────────────────────────────────────────────────
    
    "f":  _p(["consonant", "fricative", "continuant", "labial", "anterior"]),
    "v":  _p(["consonant", "fricative", "continuant", "labial", "anterior", "voiced"]),
    "th": _p(["consonant", "fricative", "continuant", "dental", "anterior",
               "coronal"]),
    "dh": _p(["consonant", "fricative", "continuant", "dental", "anterior",
               "coronal", "voiced"]),
    "s":  _p(["consonant", "fricative", "continuant", "alveolar", "anterior",
               "coronal"]),
    "z":  _p(["consonant", "fricative", "continuant", "alveolar", "anterior",
               "coronal", "voiced"]),

    "sh": _p(["consonant", "fricative", "continuant", "palatal", "posterior",
               "coronal"]),

    "zh": _p(["consonant", "fricative", "continuant", "palatal", "posterior",
               "coronal", "voiced"]),
    
    "hh": _p(["consonant", "fricative", "continuant", "glottal", "posterior",
               "dorsal"]),
 
    # ── Affricates ─────────────────────────────────────────────────────────
    
    "ch": _p(["consonant", "affricate", "palatal", "posterior", "coronal"]),
    "jh": _p(["consonant", "affricate", "palatal", "posterior", "coronal",
               "voiced"]),
 
    # ── Nasals ─────────────────────────────────────────────────────────────
    
    "m":  _p(["consonant", "sonorant", "nasal", "continuant", "labial",
               "anterior", "bilabial", "voiced"]),
    "n":  _p(["consonant", "sonorant", "nasal", "continuant", "alveolar",
               "anterior", "coronal", "voiced"]),
    "ng": _p(["consonant", "sonorant", "nasal", "continuant", "velar",
               "posterior", "dorsal", "voiced"]),
 
    # ── Liquids ────────────────────────────────────────────────────────────
    "l":  _p(["consonant", "sonorant", "approximant", "liquid", "continuant",
               "alveolar", "anterior", "coronal", "voiced"]),
    
    "r":  _p(["consonant", "sonorant", "approximant", "liquid", "continuant",
               "alveolar", "anterior", "retroflex", "coronal", "voiced"]),
 
    # ── Semivowels (Glides) ────────────────────────────────────────────────

    "w":  _p(["sonorant", "approximant", "semivowel", "continuant", "labial",
               "high", "anterior", "bilabial", "round", "voiced"]),
    "y":  _p(["sonorant", "approximant", "semivowel", "continuant", "palatal",
               "high", "posterior", "coronal", "voiced"]),
 
    # ── Short Monophthong Vowels ───────────────────────────────────────────
    "ih": _p(["sonorant", "vowel", "continuant", "high", "front",
               "short", "monophthong", "voiced"]),

    "eh": _p(["sonorant", "vowel", "mid", "front",
               "short", "monophthong", "voiced"]),

    "ae": _p(["sonorant", "vowel", "continuant", "low", "front",
               "long", "monophthong", "voiced"]),

    "ah": _p(["sonorant", "vowel", "continuant", "mid", "back",
               "short", "monophthong", "voiced"]),

    "uh": _p(["sonorant", "vowel", "continuant", "high", "back",
               "short", "monophthong", "round", "voiced"]),
 
    # ── Long Monophthong Vowels ────────────────────────────────────────────
    "iy": _p(["sonorant", "vowel", "continuant", "high", "front",
               "long", "monophthong", "voiced"]),
    "aa": _p(["sonorant", "vowel", "continuant", "low", "back",
               "long", "monophthong", "voiced"]),
    "ao": _p(["sonorant", "vowel", "continuant", "mid", "back",
               "long", "monophthong", "round", "voiced"]),
    "er": _p(["sonorant", "vowel", "continuant", "mid", "central",
               "retroflex", "short", "monophthong", "voiced"]),
    "uw": _p(["sonorant", "vowel", "continuant", "high", "back",
               "long", "monophthong", "round", "voiced"]),
 
    # ── Diphthongs ─────────────────────────────────────────────────────────
    "ey": _p(["sonorant", "vowel", "continuant", "mid", "front",
               "long", "diphthong", "voiced"]),

    "aw": _p(["sonorant", "vowel", "continuant", "low", "central",
               "long", "diphthong", "round", "voiced"]),

    "ay": _p(["sonorant", "vowel", "low", "central",
               "long", "diphthong", "voiced"]),

    "oy": _p(["sonorant", "vowel", "continuant", "mid", "back",
               "long", "diphthong", "round", "voiced"]),

    "ow": _p(["sonorant", "vowel", "continuant", "mid", "central",
               "long", "diphthong", "round", "voiced"]),
 
    # ── Silence ────────────────────────────────────────────────────────────
    # Paper: "All silence labels were further removed leaving silence frames
    # to be handled by the blank label."
    "sil": _p([]),   # all features absent; treated as blank during training
}

# Verify all 39 phonemes are covered.
# "sil" is intentionally extra — it is a fallback/blank placeholder, not a
# speech target, so it lives in PHONEME_FEATURES but not in CMU_39_PHONEMES.
_expected = set(CMU_39_PHONEMES) | {"sil"}
assert set(PHONEME_FEATURES.keys()) == _expected, (
    f"Missing from PHONEME_FEATURES : {_expected - set(PHONEME_FEATURES.keys())}\n"
    f"Unexpected in PHONEME_FEATURES: {set(PHONEME_FEATURES.keys()) - _expected}"
)
assert NUM_PHONEMES == 39, f"Expected 39 phonemes, got {NUM_PHONEMES}"


def phoneme_to_feature_vector(phoneme: str) -> list[bool]:
    """Return a binary list of length 35 for a given phoneme."""
    feat_dict = PHONEME_FEATURES.get(phoneme, PHONEME_FEATURES["sil"])
    return [feat_dict[f] for f in PHONOLOGICAL_FEATURES]


def phoneme_sequence_to_feature_sequences(
    phonemes: list[str],
) -> list[list[int]]:
    """
    Convert a phoneme sequence to N=35 binary label sequences.

    Returns:
        feature_seqs: list of 35 lists, each containing +att(1) or -att(0)
                      integers for each phoneme position.
    """
    feature_seqs = [[] for _ in range(NUM_FEATURES)]
    for ph in phonemes:
        vec = phoneme_to_feature_vector(ph)
        for feat_idx, present in enumerate(vec):
            feature_seqs[feat_idx].append(1 if present else 0)
    return feature_seqs


def feature_sequences_to_ctc_labels(
    feature_seqs: list[list[int]],
) -> list[list[int]]:
    """
    Convert binary feature sequences (0/1) to CTC label indices.

    For category i:
      - +att  →  node index i          (feature_idx_to_pos_node)
      - -att  →  node index i + 35     (feature_idx_to_neg_node)

    Returns:
        ctc_labels: list of 35 lists of node indices (int)
    """
    ctc_labels = []
    for feat_idx, seq in enumerate(feature_seqs):
        label_seq = []
        for val in seq:
            if val == 1:
                label_seq.append(feature_idx_to_pos_node(feat_idx))
            else:
                label_seq.append(feature_idx_to_neg_node(feat_idx))
        ctc_labels.append(label_seq)
    return ctc_labels
