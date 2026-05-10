"""
confusion_tables.py
===================
Reproduce Tables 5 & 6 from Shahin et al. (2025).

For each confused phoneme pair (A/B):

  Phoneme-level:
    FAR = among positions where speaker said B but canonical was A
          (or A but canonical was B) → how often did model predict canonical
    FRR = among positions where speaker correctly said A or B
          → how often did model predict something other than canonical

  Phonological-level:
    For each DISTINCTIVE feature between A and B:
      FAR_feat = same substitution positions → how often did model's
                 decoded feature value == canonical feature value
                 (i.e. wrong feature predicted — accepted the error)
      FRR_feat = correct positions → how often model's decoded feature
                 value != canonical feature value
    Report the feature with lowest FAR.

Usage:
    python confusion_tables.py \
        --l2arctic_dir /path/to/l2arctic \
        --phoneme_model /path/to/best_phoneme_model.pt \
        --phonological_model /path/to/best_model.pt \
        --speakers RRBI YBAA HJK BWC EBVS YDCK
"""

import argparse
import sys
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from alignment import levenshtein_alignment
from phonological_features import (
    PHONOLOGICAL_FEATURES, NUM_FEATURES, NUM_OUTPUT_NODES,
    BLANK_IDX, PHONEME_FEATURES,
    feature_idx_to_pos_node, feature_idx_to_neg_node,
    phoneme_to_feature_vector,
)
from dataset import normalize_phoneme


# ─────────────────────────────────────────────────────────────────────────────
# Confusion pairs from the paper (Tables 5 & 6)
# ─────────────────────────────────────────────────────────────────────────────
CONSONANT_PAIRS = [
    ("d", "dh"), ("s", "z"),  ("d", "t"),  ("b", "p"),
    ("f", "v"),  ("l", "w"),  ("t", "th"), ("g", "k"),
    ("ch", "jh"),("n", "ng"), ("v", "w"),  ("s", "th"),
    ("s", "t"),  ("s", "sh"), ("b", "v"),  ("dh", "z"),
    ("m", "n"),
]

VOWEL_PAIRS = [
    ("ih", "iy"), ("ao", "ow"), ("ah", "er"), ("eh", "ey"),
    ("ah", "ao"), ("ae", "eh"), ("aa", "ah"), ("ah", "eh"),
    ("aa", "ae"), ("aa", "aw"), ("aa", "ao"), ("ah", "ih"),
    ("ah", "ow"), ("ao", "er"), ("eh", "ih"), ("uh", "uw"),
    ("aa", "ay"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Annotation parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_annotation(textgrid_path):
    """Returns list of (canonical, human) phone pairs. human=None for deletions."""
    path = Path(textgrid_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    tier_blocks = re.split(r'item\s*\[\d+\]', content)
    phones_block = None
    for block in tier_blocks:
        if re.search(r'name\s*=\s*"phones?"', block, re.IGNORECASE):
            phones_block = block
            break
    if phones_block is None:
        return []
    intervals = re.findall(
        r'intervals\s*\[\d+\].*?xmin\s*=\s*[\d.]+.*?xmax\s*=\s*[\d.]+'
        r'.*?text\s*=\s*"([^"]*)"',
        phones_block, re.DOTALL,
    )
    pairs = []
    for text in intervals:
        text = text.strip()
        if text in ("", "sil", "sp", "spn", "<unk>"):
            continue
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 1:
            ph = normalize_phoneme(parts[0])
            if ph != "sil":
                pairs.append((ph, ph))  # correct
        elif len(parts) == 3:
            canon_raw, pron_raw, etype = parts
            etype = etype.strip().lower()
            canon = normalize_phoneme(canon_raw)
            if canon == "sil":
                continue
            if etype == "s":
                pron_clean = pron_raw.replace("*", "").strip()
                if pron_clean.lower() == "err":
                    human = None
                else:
                    human = normalize_phoneme(pron_clean)
                    if human == "sil":
                        human = None
                pairs.append((canon, human))
            elif etype == "d":
                pairs.append((canon, None))
            elif etype == "a":
                pass  # no canonical slot
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# CTC decode helpers
# ─────────────────────────────────────────────────────────────────────────────

def ctc_decode_phoneme(logits_np, blank_idx=39):
    """(T, 40) numpy → list of phoneme strings."""
    from phonological_features import CMU_39_PHONEMES
    preds = np.argmax(logits_np, axis=-1)
    phones, prev = [], -1
    for p in preds:
        if p == blank_idx:
            prev = -1
            continue
        if p != prev:
            ph = CMU_39_PHONEMES[p] if p < len(CMU_39_PHONEMES) else "sil"
            if ph != "sil":
                phones.append(ph)
            prev = p
    return phones


def ctc_decode_phonological(logits_np, valid_len=None):
    """
    (T, 71) numpy → list[35] of decoded feature sequences (each: list of 0/1).
    """
    T = valid_len if valid_len else logits_np.shape[0]
    logits_np = logits_np[:T]
    decoded = []
    for f in range(NUM_FEATURES):
        pos = feature_idx_to_pos_node(f)
        neg = feature_idx_to_neg_node(f)
        cat = logits_np[:, [pos, neg, BLANK_IDX]]
        preds = np.argmax(cat, axis=-1)
        collapsed, prev = [], -1
        for p in preds:
            if p == 2:
                prev = -1
                continue
            if p != prev:
                collapsed.append(1 if p == 0 else 0)
                prev = p
        decoded.append(collapsed)
    return decoded


def align_phones_to_canonical(predicted_phones, canonical_phones):
    """
    Levenshtein-align predicted phoneme sequence against canonical sequence.

    Returns a list of length len(canonical_phones) where each entry is the
    predicted phone aligned to that canonical slot (None = model deletion).
    Insertions (predicted phones with no canonical slot) are discarded,
    matching the author's MDD alignment logic.
    """
    if not canonical_phones:
        return []
    if not predicted_phones:
        return [None] * len(canonical_phones)

    _, _, _, ops = levenshtein_alignment(canonical_phones, predicted_phones)

    aligned = []
    hyp_cursor = 0
    for op, _ref, _hyp in ops:
        if op == "M":          # match
            aligned.append(predicted_phones[hyp_cursor])
            hyp_cursor += 1
        elif op == "S":        # substitution
            aligned.append(predicted_phones[hyp_cursor])
            hyp_cursor += 1
        elif op == "I":        # insertion in predicted — no canonical slot, skip
            hyp_cursor += 1
        elif op == "D":        # deletion in predicted — model missed this slot
            aligned.append(None)
    return aligned


def align_feature_seq_to_canonical(feat_seq, canon_binary_seq):
    """
    Levenshtein-align one decoded feature sequence (list of 0/1) against
    the canonical binary sequence for that feature.

    Returns a list of length len(canon_binary_seq) where each entry is the
    predicted feature value (0/1) or None (model deletion at that slot).
    """
    if not canon_binary_seq:
        return []
    if not feat_seq:
        return [None] * len(canon_binary_seq)

    _, _, _, ops = levenshtein_alignment(canon_binary_seq, feat_seq)

    aligned = []
    hyp_cursor = 0
    for op, _ref, _hyp in ops:
        if op in ("M", "S"):
            aligned.append(feat_seq[hyp_cursor])
            hyp_cursor += 1
        elif op == "I":        # extra predicted token — no canonical slot
            hyp_cursor += 1
        elif op == "D":        # model skipped this canonical position
            aligned.append(None)
    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair accumulator
# ─────────────────────────────────────────────────────────────────────────────

def get_distinctive_features(ph_a, ph_b):
    """Return indices of features that differ between ph_a and ph_b."""
    vec_a = phoneme_to_feature_vector(ph_a)
    vec_b = phoneme_to_feature_vector(ph_b)
    return [i for i in range(NUM_FEATURES) if vec_a[i] != vec_b[i]]


class PairStats:
    """
    Accumulates FAR/FRR counts for one (canonical_a, canonical_b) pair.

    Phoneme-level:
      substitution_positions: canonical=A, human=B (or canonical=B, human=A)
        → FAR += 1 if predicted == canonical (error accepted)
        → TR  += 1 if predicted != canonical (error detected)
      correct_positions: canonical=A, human=A (or canonical=B, human=B)
        → FRR += 1 if predicted != canonical (correct rejected)
        → TA  += 1 if predicted == canonical

    Phonological-level (per distinctive feature):
      same positions, but compare feature values.
    """
    def __init__(self, ph_a, ph_b):
        self.ph_a = ph_a
        self.ph_b = ph_b
        self.dist_feat_idxs = get_distinctive_features(ph_a, ph_b)

        # Phoneme-level counts
        self.ph_FA = 0   # substitution, model accepted (predicted canonical)
        self.ph_TR = 0   # substitution, model rejected
        self.ph_FR = 0   # correct, model rejected
        self.ph_TA = 0   # correct, model accepted

        # Phonological-level counts per distinctive feature
        self.feat_FA = defaultdict(int)
        self.feat_TR = defaultdict(int)
        self.feat_FR = defaultdict(int)
        self.feat_TA = defaultdict(int)

    def add_substitution(self, canon, pred_phone, pred_feat_vals):
        """
        canon: the canonical phoneme at this position (A or B)
        pred_phone: what the phoneme model predicted
        pred_feat_vals: list[35] of predicted feature values (0/1/None)
        """
        # Phoneme-level
        if pred_phone == canon:
            self.ph_FA += 1
        else:
            self.ph_TR += 1

        # Phonological-level
        canon_vec = phoneme_to_feature_vector(canon)
        for f_idx in self.dist_feat_idxs:
            canon_f = 1 if canon_vec[f_idx] else 0
            pred_f = pred_feat_vals[f_idx]
            if pred_f is None:
                # Model produced nothing (shorter decode) → treat as wrong
                self.feat_TR[f_idx] += 1
            elif pred_f == canon_f:
                self.feat_FA[f_idx] += 1
            else:
                self.feat_TR[f_idx] += 1

    def add_correct(self, canon, pred_phone, pred_feat_vals):
        """Correct pronunciation position."""
        # Phoneme-level
        if pred_phone == canon:
            self.ph_TA += 1
        else:
            self.ph_FR += 1

        # Phonological-level
        canon_vec = phoneme_to_feature_vector(canon)
        for f_idx in self.dist_feat_idxs:
            canon_f = 1 if canon_vec[f_idx] else 0
            pred_f = pred_feat_vals[f_idx]
            if pred_f is None:
                self.feat_FR[f_idx] += 1
            elif pred_f == canon_f:
                self.feat_TA[f_idx] += 1
            else:
                self.feat_FR[f_idx] += 1

    def phoneme_FAR(self):
        d = self.ph_FA + self.ph_TR
        return self.ph_FA / d * 100 if d > 0 else float("nan")

    def phoneme_FRR(self):
        d = self.ph_FR + self.ph_TA
        return self.ph_FR / d * 100 if d > 0 else float("nan")

    def best_feature(self):
        """Return (feat_name, FAR, FRR) for the distinctive feature with lowest FAR."""
        best_far = float("inf")
        best = None
        for f_idx in self.dist_feat_idxs:
            fa = self.feat_FA[f_idx]
            tr = self.feat_TR[f_idx]
            fr = self.feat_FR[f_idx]
            ta = self.feat_TA[f_idx]
            far = fa / (fa + tr) * 100 if (fa + tr) > 0 else float("nan")
            frr = fr / (fr + ta) * 100 if (fr + ta) > 0 else float("nan")
            if not np.isnan(far) and far < best_far:
                best_far = far
                best = (PHONOLOGICAL_FEATURES[f_idx], far, frr)
        return best  # (feature_name, FAR%, FRR%)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_confusion_analysis(
    l2arctic_dir, speakers,
    phoneme_model, phonological_model,
    feature_extractor, device,
):
    import torch, torchaudio

    # Build all pair stats
    all_pairs = CONSONANT_PAIRS + VOWEL_PAIRS
    pair_stats = {}
    for a, b in all_pairs:
        pair_stats[(a, b)] = PairStats(a, b)

    root = Path(l2arctic_dir)

    for spk in speakers:
        ann_dir = root / spk / "annotation"
        wav_dir = root / spk / "wav"
        if not ann_dir.exists():
            print(f"  Skipping {spk}: no annotation dir")
            continue

        for ann_file in sorted(ann_dir.glob("*.TextGrid")):
            utt_id   = ann_file.stem
            wav_file = wav_dir / f"{utt_id}.wav"
            if not wav_file.exists():
                continue

            pairs = parse_annotation(str(ann_file))
            if not pairs:
                continue

            # canonical sequence — what should have been said at each position
            canonical_seq = [canon for canon, _ in pairs]

            # Load audio
            waveform, sr = torchaudio.load(str(wav_file))
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.mean(dim=0)

            inputs = feature_extractor(
                waveform.numpy(), sampling_rate=16000,
                return_tensors="pt", padding=True,
            )

            # ── Phoneme model ────────────────────────────────────────────────
            with torch.no_grad():
                ph_logits, ph_lengths = phoneme_model(
                    inputs.input_values.to(device),
                    inputs.attention_mask.to(device),
                )
            ph_logits_np = ph_logits.squeeze(0).cpu().numpy()
            ph_len = int(ph_lengths[0].item())
            predicted_phones = ctc_decode_phoneme(ph_logits_np[:ph_len])

            # Levenshtein-align predicted phones against canonical sequence.
            # Each canonical slot gets the aligned predicted phone (None = deletion).
            ph_aligned = align_phones_to_canonical(predicted_phones, canonical_seq)

            # ── Phonological model ───────────────────────────────────────────
            with torch.no_grad():
                feat_logits, feat_lengths = phonological_model(
                    inputs.input_values.to(device),
                    inputs.attention_mask.to(device),
                )
            feat_logits_np = feat_logits.squeeze(0).cpu().numpy()
            feat_len = int(feat_lengths[0].item())
            decoded_feats = ctc_decode_phonological(feat_logits_np, feat_len)

            # For each feature, build the canonical binary vector and
            # Levenshtein-align the decoded feature sequence against it.
            # This correctly handles length mismatches per feature independently.
            feat_aligned = []
            for f_idx in range(NUM_FEATURES):
                canon_feat_seq = [
                    1 if phoneme_to_feature_vector(c)[f_idx] else 0
                    for c in canonical_seq
                ]
                aligned_f = align_feature_seq_to_canonical(
                    decoded_feats[f_idx], canon_feat_seq
                )
                feat_aligned.append(aligned_f)

            # ── Score each canonical position ────────────────────────────────
            for i, (canon, human) in enumerate(pairs):
                pred_phone    = ph_aligned[i]
                pred_feat_vals = [feat_aligned[f][i] for f in range(NUM_FEATURES)]

                for (a, b), stats in pair_stats.items():
                    is_sub_ab    = (canon == a and human == b)
                    is_sub_ba    = (canon == b and human == a)
                    is_correct_a = (canon == a and human == a)
                    is_correct_b = (canon == b and human == b)

                    if is_sub_ab or is_sub_ba:
                        stats.add_substitution(canon, pred_phone, pred_feat_vals)
                    elif is_correct_a or is_correct_b:
                        stats.add_correct(canon, pred_phone, pred_feat_vals)

    return pair_stats


def _bold(s):
    """Wrap string in ANSI bold for terminal output."""
    return f"\033[1m{s}\033[0m"


def print_table(pair_list, pair_stats, title):
    """
    Print a table matching Tables 5 & 6 in Shahin et al. (2025).

    Columns: Conf_ph | Phonetic FAR | Phonetic FRR | Feature | Phonological FAR | Phonological FRR

    The lower value in each FAR/FRR pair (phonetic vs phonological) is bolded,
    matching the paper's bold convention.
    """
    SEP = "─" * 68
    HDR = "=" * 68

    print(f"\n{HDR}")
    print(f"  {title}")
    print(f"{HDR}")
    print(f"  {'Conf_ph':<10}  {'Phonetic':^17}    {'':^12}  {'Phonological':^17}")
    print(f"  {'':10}  {'FAR':>7}  {'FRR':>7}    {'Feature':<12}  {'FAR':>7}  {'FRR':>7}")
    print(f"  {SEP}")

    far_improvements = []
    for a, b in pair_list:
        key = (a, b)
        stats = pair_stats.get(key)
        if stats is None:
            continue

        ph_far = stats.phoneme_FAR()
        ph_frr = stats.phoneme_FRR()
        best   = stats.best_feature()
        conf_ph = f"{a}/{b}"

        if best is None:
            far_s = f"{ph_far:7.2f}" if not np.isnan(ph_far) else "    n/a"
            frr_s = f"{ph_frr:7.2f}" if not np.isnan(ph_frr) else "    n/a"
            print(f"  {conf_ph:<10}  {far_s}  {frr_s}    {'—':<12}  {'':>7}  {'':>7}")
            continue

        feat_name, feat_far, feat_frr = best

        # Bold the lower FAR
        if not (np.isnan(ph_far) or np.isnan(feat_far)):
            if feat_far <= ph_far:
                ph_far_s, feat_far_s = f"{ph_far:7.2f}", _bold(f"{feat_far:7.2f}")
            else:
                ph_far_s, feat_far_s = _bold(f"{ph_far:7.2f}"), f"{feat_far:7.2f}"
        else:
            ph_far_s   = f"{ph_far:7.2f}"   if not np.isnan(ph_far)   else "    n/a"
            feat_far_s = f"{feat_far:7.2f}" if not np.isnan(feat_far) else "    n/a"

        # Bold the lower FRR
        if not (np.isnan(ph_frr) or np.isnan(feat_frr)):
            if feat_frr <= ph_frr:
                ph_frr_s, feat_frr_s = f"{ph_frr:7.2f}", _bold(f"{feat_frr:7.2f}")
            else:
                ph_frr_s, feat_frr_s = _bold(f"{ph_frr:7.2f}"), f"{feat_frr:7.2f}"
        else:
            ph_frr_s   = f"{ph_frr:7.2f}"   if not np.isnan(ph_frr)   else "    n/a"
            feat_frr_s = f"{feat_frr:7.2f}" if not np.isnan(feat_frr) else "    n/a"

        if not (np.isnan(ph_far) or np.isnan(feat_far)):
            far_improvements.append(ph_far - feat_far)

        print(f"  {conf_ph:<10}  {ph_far_s}  {ph_frr_s}    {feat_name:<12}  {feat_far_s}  {feat_frr_s}")

    print(f"  {SEP}")
    if far_improvements:
        n_better = sum(1 for x in far_improvements if x > 0)
        print(f"\n  Phonological FAR lower in {n_better}/{len(pair_list)} pairs  "
              f"(avg improvement: {np.mean(far_improvements):+.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2arctic_dir",        required=True)
    parser.add_argument("--phoneme_model",        required=True)
    parser.add_argument("--phonological_model",   required=True)
    parser.add_argument("--feature_extractor",    default="facebook/wav2vec2-large-robust")
    parser.add_argument("--speakers", nargs="+",
                        default=["RRBI", "YBAA", "HJK", "BWC", "EBVS", "YDCK"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from transformers import Wav2Vec2FeatureExtractor
    from wav2vec2_phonological import PhonemeLevelWav2Vec2, PhonologicalWav2Vec2

    device = args.device
    print(f"Speakers: {args.speakers}")
    print(f"Device:   {device}\n")

    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.feature_extractor)

    print("Loading phoneme model...")
    ph_model = PhonemeLevelWav2Vec2()
    state = torch.load(args.phoneme_model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    ph_model.load_state_dict(state)
    ph_model.to(device).eval()

    print("Loading phonological model...")
    feat_model = PhonologicalWav2Vec2()
    state = torch.load(args.phonological_model, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    feat_model.load_state_dict(state)
    feat_model.to(device).eval()

    print("\nRunning analysis...\n")
    pair_stats = run_confusion_analysis(
        args.l2arctic_dir, args.speakers,
        ph_model, feat_model, fe, device,
    )

    print_table(CONSONANT_PAIRS, pair_stats, "Table 5 — Consonant Confusion Pairs")
    print_table(VOWEL_PAIRS,     pair_stats, "Table 6 — Vowel Confusion Pairs")


if __name__ == "__main__":
    main()
