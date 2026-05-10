"""
confusion_tables.py
===================
Reproduce Tables 5 & 6 from Shahin et al. (2025).

Alignment: predicted vs HUMAN sequence (Levenshtein), exactly as in
mdd_evaluation.py — NOT vs canonical.

Usage:
    python confusion_tables.py \
        --l2arctic_dir /path/to/l2arctic \
        --phoneme_model /path/to/best_phoneme_model.pt \
        --phonological_model /path/to/best_model.pt \
        --speakers RRBI YBAA HJK BWC EBVS YDCK
"""

import argparse
import sys
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
from mdd_evaluation import (
    parse_annotation_for_mdd,
    _decode_sctcSB_logits_to_feature_sequences,
    _phoneme_to_binary_array,
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
# Levenshtein alignment — mirrors count_phoneme_mdd Step 1 exactly
# ─────────────────────────────────────────────────────────────────────────────

def _compute_alignment(human, predicted):
    """
    Align predicted against human sequence using Levenshtein, exactly
    as in count_phoneme_mdd Step 1.

    Returns:
        asr_evl        : list[H] 'hit' | 'replace' | 'delete'
        hyp_pos_arr    : list[H] index into predicted (-1 if delete)
        asr_ins_errors : list of (ref_cursor, hyp_pos) for model insertions
    """
    H = len(human)
    if H == 0:
        return [], [], []
    if not predicted:
        return ["delete"] * H, [-1] * H, []

    _, _, _, ops = levenshtein_alignment(human, predicted)

    asr_evl        = ["hit"] * H
    hyp_pos_arr    = list(range(H))
    asr_ins_errors = []
    hyp_offset     = 0
    ref_cursor     = 0

    for op, _ref, _hyp in ops:
        if op == "I":
            asr_ins_errors.append((ref_cursor, ref_cursor + hyp_offset))
            for j in range(ref_cursor, H):
                hyp_pos_arr[j] += 1
            hyp_offset += 1
        elif op == "D":
            asr_evl[ref_cursor]     = "delete"
            hyp_pos_arr[ref_cursor] = -1
            for j in range(ref_cursor + 1, H):
                hyp_pos_arr[j] -= 1
            hyp_offset -= 1
            ref_cursor += 1
        else:
            if op == "S":
                asr_evl[ref_cursor] = "replace"
            ref_cursor += 1

    return asr_evl, hyp_pos_arr, asr_ins_errors


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair accumulator
# ─────────────────────────────────────────────────────────────────────────────

def _get_distinctive_features(ph_a, ph_b):
    vec_a = phoneme_to_feature_vector(ph_a)
    vec_b = phoneme_to_feature_vector(ph_b)
    return [i for i in range(NUM_FEATURES) if vec_a[i] != vec_b[i]]


class PairStats:
    """
    Accumulates FAR/FRR for one (a, b) confusion pair using the same
    asr_evl x error classification as count_phoneme_mdd /
    count_phonological_mdd.

    Substitution positions (error='s', canon in {a,b}, human is the other):
        Phoneme:  hit or replace → FA  (model accepted the error)
                  delete         → TR  (model detected something wrong)
        Feature:  same as count_phonological_mdd with cma check

    Correct positions (error='c', canon=human in {a,b}):
        Phoneme:  hit     → TA
                  replace → FR
                  delete  → FR
        Feature:  same as count_phonological_mdd (cma always True for error='c')
    """
    def __init__(self, ph_a, ph_b):
        self.ph_a = ph_a
        self.ph_b = ph_b
        self.dist_feat_idxs = _get_distinctive_features(ph_a, ph_b)

        self.ph_FA = 0
        self.ph_TR = 0
        self.ph_FR = 0
        self.ph_TA = 0

        self.feat_FA = defaultdict(int)
        self.feat_TR = defaultdict(int)
        self.feat_FR = defaultdict(int)
        self.feat_TA = defaultdict(int)

    def add_phoneme_substitution(self, evl):
        if evl in ("hit", "replace"):
            self.ph_FA += 1
        else:
            self.ph_TR += 1

    def add_phoneme_correct(self, evl):
        if evl == "hit":
            self.ph_TA += 1
        else:
            self.ph_FR += 1

    def add_feat_substitution(self, evl, f_idx, cma):
        """Mirrors count_phonological_mdd Step 2 for error='s'."""
        if evl == "hit":
            if cma:  self.feat_TA[f_idx] += 1
            else:    self.feat_TR[f_idx] += 1
        elif evl == "replace":
            if cma:  self.feat_FR[f_idx] += 1
            else:    self.feat_FA[f_idx] += 1
        elif evl == "delete":
            if cma:  self.feat_FR[f_idx] += 1
            else:    self.feat_TR[f_idx] += 1

    def add_feat_correct(self, evl, f_idx):
        """error='c' → cma is always True."""
        if evl == "hit":
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
        """Return (feat_name, FAR%, FRR%) for the distinctive feature with lowest FAR."""
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
        return best


# ─────────────────────────────────────────────────────────────────────────────
# CTC decode
# ─────────────────────────────────────────────────────────────────────────────

def ctc_decode_phoneme(logits_np, blank_idx=39):
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


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_confusion_analysis(
    l2arctic_dir, speakers,
    phoneme_model, phonological_model,
    feature_extractor, device,
):
    import torch, torchaudio

    all_pairs = CONSONANT_PAIRS + VOWEL_PAIRS
    pair_stats = {(a, b): PairStats(a, b) for a, b in all_pairs}

    # For each canonical phone, which pairs need scoring?
    pairs_for_phone = defaultdict(list)
    for (a, b) in all_pairs:
        pairs_for_phone[a].append((a, b))
        pairs_for_phone[b].append((a, b))

    root = Path(l2arctic_dir)

    for spk in speakers:
        ann_dir = root / spk / "annotation"
        wav_dir = root / spk / "wav"
        if not ann_dir.exists():
            print(f"  Skipping {spk}: no annotation dir")
            continue

        ann_files = sorted(ann_dir.glob("*.TextGrid"))
        print(f"  [{spk}] {len(ann_files)} utterances")

        for ann_file in ann_files:
            utt_id   = ann_file.stem
            wav_file = wav_dir / f"{utt_id}.wav"
            if not wav_file.exists():
                continue

            # ── Annotation ─────────────────────────────────────────────────
            (human, canonical, pron_errors,
             exp_trans, act_trans, ori_indx) = parse_annotation_for_mdd(str(ann_file))
            if not human:
                continue

            H     = len(human)
            n_ann = len(pron_errors)

            # Per-interval feature vectors (for cma check)
            canon_feats_by_ori  = np.zeros((n_ann, NUM_FEATURES), dtype=np.int8)
            actual_feats_by_ori = np.zeros((n_ann, NUM_FEATURES), dtype=np.int8)
            for ann_i in range(n_ann):
                canon_feats_by_ori[ann_i]  = _phoneme_to_binary_array(exp_trans[ann_i])
                actual_feats_by_ori[ann_i] = _phoneme_to_binary_array(act_trans[ann_i])

            # ── Audio ──────────────────────────────────────────────────────
            waveform, sr = torchaudio.load(str(wav_file))
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.mean(dim=0)

            inputs = feature_extractor(
                waveform.numpy(), sampling_rate=16000,
                return_tensors="pt", padding=True,
            )

            # ── Phoneme model → align vs HUMAN ─────────────────────────────
            with torch.no_grad():
                ph_logits, ph_lengths = phoneme_model(
                    inputs.input_values.to(device),
                    inputs.attention_mask.to(device),
                )
            ph_logits_np = ph_logits.squeeze(0).cpu().numpy()
            ph_len       = int(ph_lengths[0].item())
            predicted_ph = ctc_decode_phoneme(ph_logits_np[:ph_len])

            ph_evl, _, _ = _compute_alignment(human, predicted_ph)

            # ── Phonological model → per-feature align vs HUMAN binary ─────
            with torch.no_grad():
                feat_logits, feat_lengths = phonological_model(
                    inputs.input_values.to(device),
                    inputs.attention_mask.to(device),
                )
            feat_logits_np = feat_logits.squeeze(0).cpu().numpy()
            feat_len       = int(feat_lengths[0].item())
            pred_feat_seqs = _decode_sctcSB_logits_to_feature_sequences(
                feat_logits_np[:feat_len]
            )

            human_feats = np.stack(      # (H, 35)
                [_phoneme_to_binary_array(ph) for ph in human]
            )

            # Align each feature sequence vs human binary (same as
            # count_phonological_mdd Step 1, one per feature)
            feat_evl_all = []
            for f_idx in range(NUM_FEATURES):
                human_binary = human_feats[:, f_idx].astype(int).tolist()
                pred_binary  = pred_feat_seqs[f_idx]
                f_evl, _, _  = _compute_alignment(human_binary, pred_binary)
                feat_evl_all.append(f_evl)

            # ── Score each human-sequence position ─────────────────────────
            for ref_pos, ori_pos in zip(range(H), ori_indx):
                error  = pron_errors[ori_pos]
                canon  = exp_trans[ori_pos]
                actual = act_trans[ori_pos]

                if error not in ("c", "s"):
                    continue

                relevant = pairs_for_phone.get(canon, [])
                if not relevant:
                    continue

                ph_ev = ph_evl[ref_pos]

                for (a, b) in relevant:
                    stats = pair_stats[(a, b)]
                    other = b if canon == a else a

                    if error == "s" and actual != other:
                        # Substitution but human said something outside this pair
                        continue

                    # ── Phoneme-level ──────────────────────────────────────
                    if error == "s":
                        stats.add_phoneme_substitution(ph_ev)
                    else:
                        stats.add_phoneme_correct(ph_ev)

                    # ── Phonological-level ─────────────────────────────────
                    for f_idx in stats.dist_feat_idxs:
                        f_ev = feat_evl_all[f_idx][ref_pos]
                        cf   = int(canon_feats_by_ori[ori_pos, f_idx])
                        af   = int(actual_feats_by_ori[ori_pos, f_idx])
                        cma  = (cf == af)
                        if error == "s":
                            stats.add_feat_substitution(f_ev, f_idx, cma)
                        else:
                            stats.add_feat_correct(f_ev, f_idx)

    return pair_stats


# ─────────────────────────────────────────────────────────────────────────────
# Table printer
# ─────────────────────────────────────────────────────────────────────────────

def _bold(s):
    return f"\033[1m{s}\033[0m"


def print_table(pair_list, pair_stats, title):
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
        stats   = pair_stats.get((a, b))
        if stats is None:
            continue
        ph_far  = stats.phoneme_FAR()
        ph_frr  = stats.phoneme_FRR()
        best    = stats.best_feature()
        conf_ph = f"{a}/{b}"

        if best is None:
            far_s = f"{ph_far:7.2f}" if not np.isnan(ph_far) else "    n/a"
            frr_s = f"{ph_frr:7.2f}" if not np.isnan(ph_frr) else "    n/a"
            print(f"  {conf_ph:<10}  {far_s}  {frr_s}    {'—':<12}  {'':>7}  {'':>7}")
            continue

        feat_name, feat_far, feat_frr = best

        if not (np.isnan(ph_far) or np.isnan(feat_far)):
            if feat_far <= ph_far:
                ph_far_s, feat_far_s = f"{ph_far:7.2f}", _bold(f"{feat_far:7.2f}")
            else:
                ph_far_s, feat_far_s = _bold(f"{ph_far:7.2f}"), f"{feat_far:7.2f}"
        else:
            ph_far_s   = f"{ph_far:7.2f}"   if not np.isnan(ph_far)   else "    n/a"
            feat_far_s = f"{feat_far:7.2f}" if not np.isnan(feat_far) else "    n/a"

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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

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
