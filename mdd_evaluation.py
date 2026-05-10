"""
mdd_evaluation.py
=================
Mispronunciation Detection and Diagnosis (MDD) evaluation.

Supports two evaluation levels:
  1. Phoneme-level MDD      -- PhonemeLevelWav2Vec2 (39 phonemes + blank, CTC)
  2. Phonological-level MDD -- PhonologicalWav2Vec2 SCTC-SB (71-node output)

Both models share the same Wav2Vec2FeatureExtractor.

Metrics:
  FAR = FA / (FA + TR)     False Acceptance Rate
  FRR = FR / (FR + TA)     False Rejection Rate
  DER = DE / (CD + DE)     Diagnostic Error Rate

MDD categories (per canonical position):
  TA    : predicted == canonical  AND  human == canonical
  FA    : predicted == canonical  AND  human != canonical
  FR    : predicted != canonical  AND  human == canonical
  TR/CD : predicted != canonical  AND  human != canonical  AND  predicted == human
  TR/DE : predicted != canonical  AND  human != canonical  AND  predicted != human

  NOTE — phoneme vs phonological level differ for substitution cases:

  Phoneme level (count_phoneme_mdd):
    hit    + s → always TR+CD  (no cma check; Shahin's phoneme evaluator)
    replace+ s → always FA
    delete + s → always TR+DE

  Phonological level (count_phonological_mdd):
    hit    + s → TA    if canon feature == actual feature (cma),  else TR+CD
    replace+ s → FR    if canon feature == actual feature (cma),  else FA
    delete + s → FR    if canon feature == actual feature (cma),  else TR+DE
    (cma = the substitution did not change this particular feature)

  None values (deletion gaps from Levenshtein alignment) are treated as
  distinct from every phoneme/feature value, so:

    human=None means speaker deleted the canonical phoneme.
    pred=None  means model emitted nothing aligned to this canonical slot.

  None is never equal to a real phoneme/feature value, which gives:

    human == canonical, pred == canonical  → TA   (impossible if pred=None)
    human != canonical, pred == canonical  → FA   (impossible if pred=None)
    human == canonical, pred != canonical  → FR   (covers pred=None: model
                                                    wrongly rejected a correct
                                                    pronunciation or a deletion
                                                    the speaker made correctly)
    human != canonical, pred != canonical:
      pred == human                        → TR/CD (both speaker and model
                                                    deleted/substituted the same
                                                    way; diagnosis correct)
      pred != human                        → TR/DE (error detected but model
                                                    output does not match what
                                                    speaker actually said)

  Concretely, deletion cases follow the same rules as substitutions
  (mirroring ins_del_sub_cor_analysis.py):
    human=None, pred=None   → TR/CD  (del_del: both deleted — correct diagnosis)
    human=None, pred=canon  → FA     (del_nodel: speaker deleted, model accepted)
    human=None, pred=other  → TR/DE  (del_del1: speaker deleted, model wrong)

The three sequences used for evaluation
-----------------------------------------
  canonical
      Derived directly from annotation/<utt_id>.TextGrid:
      the canonical field of each interval (first element of comma-separated label
      for error intervals; the label itself for correct intervals).

  human-annotated
      Derived directly from the same annotation TextGrid:
      what the speaker actually produced (substituted phones used, deletions
      omitted, insertions included).

  predicted
      Phoneme-level : PhonemeLevelWav2Vec2 logits (T, 40) → greedy CTC decode
                      → list[str] of CMU-39 phonemes
      Phonological  : PhonologicalWav2Vec2 logits (T, 71) numpy array


Alignment
---------
canonical ↔ human:
  human is built directly from the annotation with one slot per canonical
  phoneme. Deletions append None, additions are discarded (no canonical
  anchor). This keeps human length-matched to canonical — simple zip is safe.

canonical ↔ predicted (phoneme-level):
  The model's CTC-decoded phoneme sequence has its own insertions and
  deletions. Levenshtein alignment is used to map predicted phones to
  canonical positions. Model insertions (no canonical slot) are discarded.
  Model deletions (missed canonical phone) become None at that position.

canonical ↔ predicted (phonological-level):
  After CTC collapse, each of the 35 feature sequences has its own length
  U_f ≤ T. Each feature sequence is independently Levenshtein-aligned to
  its corresponding canonical binary sequence (length N). This avoids the
  zip pitfall where U ≠ N would silently misalign every subsequent position.

Phonological-level evaluation (wav2vec2)
-----------------------------------------
Evaluation is performed for each of the 35 features independently.
For each feature f:
  1. canonical phoneme sequence  →  N binary values  (1=+att, 0=-att)
  2. human phoneme sequence      →  N binary values  (aligned to canonical)
  3. logits (T, 71)              →  per-feature CTC collapse → U_f binary values
                                  →  Levenshtein-aligned to N canonical positions
                                     (model deletions → None at that position)
Each position of each feature is then classified TA/FA/FR/TR_CD/TR_DE.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from phonological_features import (
    PHONOLOGICAL_FEATURES,
    NUM_FEATURES,
    BLANK_IDX,
    NUM_OUTPUT_NODES,
    PHONEME_FEATURES,
    feature_idx_to_pos_node,
    feature_idx_to_neg_node,
)
from dataset import normalize_phoneme

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Annotation TextGrid parser — matches author's data layout exactly
# ─────────────────────────────────────────────────────────────────────────────

def parse_annotation_for_mdd(
    textgrid_path: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[int]]:
    """
    Parse an L2-ARCTIC annotation TextGrid and return all structures needed
    for MDD evaluation, matching the author's data layout exactly.

    TextGrid phones tier label formats
    -----------------------------------
      Correct:      "AH1"        error='c'  canon=ah   actual=ah
      Substitution: "DH,D,s"    error='s'  canon=dh   actual=d
      Deletion:     "TH,sil,d"  error='d'  canon=th   actual=sil (speaker omitted)
      Addition:     "sil,AH,a"  error='a'  canon=sil  actual=ah  (extra phone)
      Silence:      ""/"sil"    skip entirely

    The HUMAN (actual) sequence is the alignment reference — matching the
    author who aligns model output against what was actually said, then uses
    error labels to classify outcomes.

    Addition errors ARE included in the human sequence (the speaker produced
    them), with error='a' in pron_errors and 'sil' in exp_trans.

    Deletion errors are NOT in the human sequence (speaker said nothing),
    but they remain in pron_errors / exp_trans for the two-pass deletion
    matching performed in count_*_mdd.

    Returns
    -------
    human      : list[str]  phones the speaker actually produced (no deletions,
                             additions included). CTC alignment reference.
    canonical  : list[str]  canonical phone per human-sequence position
                             (exp_trans[ori_indx[i]] for each i).
    pron_errors: list[str]  per-annotation-interval error type ('c','s','d','a').
                             Includes deletion intervals absent from human.
    exp_trans  : list[str]  normalised canonical phone per annotation interval.
                             'sil' for addition intervals (no canonical phone).
    act_trans  : list[str]  normalised actual phone per annotation interval.
                             'sil' for deletion intervals (speaker said nothing).
    ori_indx   : list[int]  maps human[i] → index in pron_errors/exp_trans/act_trans.
    """
    path = Path(textgrid_path)
    if not path.exists():
        return [], [], [], [], [], []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    tier_blocks = re.split(r'item\s*\[\d+\]', content)
    phones_block = None
    for block in tier_blocks:
        if re.search(r'name\s*=\s*"phones?"', block, re.IGNORECASE):
            phones_block = block
            break
    if phones_block is None:
        return [], [], [], [], [], []

    intervals = re.findall(
        r'intervals\s*\[\d+\].*?xmin\s*=\s*[\d.]+.*?xmax\s*=\s*[\d.]+'
        r'.*?text\s*=\s*"([^"]*)"',
        phones_block, re.DOTALL,
    )

    pron_errors: list[str] = []
    exp_trans:   list[str] = []
    act_trans:   list[str] = []
    human:       list[str] = []
    ori_indx:    list[int] = []
    ann_idx = 0

    for text in intervals:
        text = text.strip()
        if text in ("", "sil", "sp", "spn", "<unk>"):
            continue

        parts = [p.strip() for p in text.split(",")]

        if len(parts) == 1:
            ph = normalize_phoneme(parts[0])
            if ph == "sil":
                continue
            pron_errors.append("c")
            exp_trans.append(ph)
            act_trans.append(ph)
            human.append(ph)
            ori_indx.append(ann_idx)
            ann_idx += 1

        elif len(parts) == 3:
            canonical_raw, pronounced_raw, error_type = parts
            error_type = error_type.strip().lower()

            if error_type == "s":
                canon_ph = normalize_phoneme(canonical_raw)
                if canon_ph == "sil":
                    continue
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                act_ph = ("sil" if pronounced_clean.lower() == "err"
                          else normalize_phoneme(pronounced_clean))
                pron_errors.append("s")
                exp_trans.append(canon_ph)
                act_trans.append(act_ph)
                if act_ph != "sil":
                    human.append(act_ph)
                    ori_indx.append(ann_idx)
                ann_idx += 1

            elif error_type == "d":
                # Deletion — kept in pron_errors for two-pass matching,
                # NOT added to human (speaker said nothing).
                canon_ph = normalize_phoneme(canonical_raw)
                if canon_ph == "sil":
                    continue
                pron_errors.append("d")
                exp_trans.append(canon_ph)
                act_trans.append("sil")
                ann_idx += 1

            elif error_type == "a":
                # Addition — included in human (speaker said it).
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                act_ph = normalize_phoneme(pronounced_clean)
                if act_ph == "sil":
                    continue
                pron_errors.append("a")
                exp_trans.append("sil")
                act_trans.append(act_ph)
                human.append(act_ph)
                ori_indx.append(ann_idx)
                ann_idx += 1

    canonical = [exp_trans[ori_indx[i]] for i in range(len(human))]
    return human, canonical, pron_errors, exp_trans, act_trans, ori_indx

# ─────────────────────────────────────────────────────────────────────────────
# Alignment helper
# ─────────────────────────────────────────────────────────────────────────────

def _zip_to_canonical(canonical: list, other: list) -> list:
    """
    Align `other` to `canonical` position-by-position.

    Returns a list of length len(canonical). Positions where `other` is shorter
    are filled with None (deletion). Tokens beyond len(canonical) are ignored.
    """
    return [other[i] if i < len(other) else None for i in range(len(canonical))]

# ─────────────────────────────────────────────────────────────────────────────
# MDD count dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MDDCounts:
    """Raw MDD counts for one or many utterances at phoneme or feature level."""
    TA:    int = 0
    FA:    int = 0
    FR:    int = 0
    TR_CD: int = 0
    TR_DE: int = 0

    @property
    def TR(self) -> int:
        return self.TR_CD + self.TR_DE

    @property
    def FAR(self) -> float:
        """False Acceptance Rate = FA / (FA + TR)"""
        d = self.FA + self.TR
        return self.FA / d if d > 0 else 0.0

    @property
    def FRR(self) -> float:
        """False Rejection Rate = FR / (FR + TA)"""
        d = self.FR + self.TA
        return self.FR / d if d > 0 else 0.0

    @property
    def DER(self) -> float:
        """Diagnostic Error Rate = DE / (CD + DE)"""
        d = self.TR_CD + self.TR_DE
        return self.TR_DE / d if d > 0 else 0.0

    def __add__(self, other: "MDDCounts") -> "MDDCounts":
        return MDDCounts(
            TA    = self.TA    + other.TA,
            FA    = self.FA    + other.FA,
            FR    = self.FR    + other.FR,
            TR_CD = self.TR_CD + other.TR_CD,
            TR_DE = self.TR_DE + other.TR_DE,
        )

    def summary(self) -> dict:
        return {
            "TA": self.TA, "FA": self.FA, "FR": self.FR,
            "TR": self.TR, "TR_CD": self.TR_CD, "TR_DE": self.TR_DE,
            "FAR": round(self.FAR, 4),
            "FRR": round(self.FRR, 4),
            "DER": round(self.DER, 4),
        }


@dataclass
class PhonologicalMDDCounts:
    """One MDDCounts per phonological feature (35 total) for wav2vec2 evaluation."""
    counts: dict[str, MDDCounts] = field(
        default_factory=lambda: {f: MDDCounts() for f in PHONOLOGICAL_FEATURES}
    )

    def add(self, other: "PhonologicalMDDCounts"):
        for feat in PHONOLOGICAL_FEATURES:
            self.counts[feat] = self.counts[feat] + other.counts[feat]

    def summary(self) -> dict:
        out = {feat: cnt.summary() for feat, cnt in self.counts.items()}
        out["__macro_avg__"] = {
            "FAR": round(float(np.mean([c.FAR for c in self.counts.values()])), 4),
            "FRR": round(float(np.mean([c.FRR for c in self.counts.values()])), 4),
            "DER": round(float(np.mean([c.DER for c in self.counts.values()])), 4),
        }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Core counting: phoneme-level
# ─────────────────────────────────────────────────────────────────────────────

def count_phoneme_mdd(
    human:       list[str],
    canonical:   list[str],
    pron_errors: list[str],
    exp_trans:   list[str],
    act_trans:   list[str],
    ori_indx:    list[int],
    predicted:   list[str],
) -> MDDCounts:
    """
    Phoneme-level MDD counting for a single utterance.

    Matches the author's algorithm exactly:

    Step 1 — Levenshtein-align predicted against HUMAN (actual) sequence.
    Step 2 — Main loop: classify each (ref_pos, asr_pos, ori_pos) by
              error type and alignment result.
    Step 3 — Model insertion loop: insertions adjacent to 'd' errors →
              FA or TR/DE; all others → FR.
    Step 4 — Unmatched deletion cleanup: 'd' errors not consumed → TR/CD.

    Classification (asr_evl × error):
      hit    + c → TA
      hit    + s → TR+CD  (model agrees with human error — correct diagnosis)
      hit    + a → TR+CD
      replace+ c → FR
      replace+ s → FA     (model predicts canonical despite human error)
      replace+ a → TR+DE
      delete + c → FR
      delete + s → TR+DE  (model missed the phone, human error present)
      delete + a → FA
    """
    from alignment import levenshtein_alignment

    counts = MDDCounts()
    H = len(human)
    if H == 0:
        return counts

    # Step 1: Levenshtein-align predicted against HUMAN
    _, _, _, ops = levenshtein_alignment(human, predicted)

    asr_evl     = ["hit"] * H
    hyp_pos_arr = list(range(H))
    asr_ins_errors: list[tuple[int, int]] = []
    hyp_offset  = 0
    ref_cursor  = 0

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

    # Step 2: main classification loop
    handeled_del: list[int] = []

    for ref_pos, asr_pos, ori_pos in zip(range(H), hyp_pos_arr, ori_indx):
        evl   = asr_evl[ref_pos]
        error = pron_errors[ori_pos]

        if evl == "hit":
            if error == "c":             counts.TA    += 1
            elif error == "s":           counts.TR_CD += 1  # always TR+CD at phoneme level
            elif error == "a":           counts.TR_CD += 1

        elif evl == "replace":
            if error == "c":             counts.FR    += 1
            elif error == "s":           counts.FA    += 1  # model predicts canon, human wrong → FA
            elif error == "a":           counts.TR_DE += 1

        elif evl == "delete":
            if error == "c":             counts.FR    += 1
            elif error == "s":           counts.TR_DE += 1  # always TR+DE at phoneme level
            elif error == "a":           counts.FA    += 1

    # Step 3: model insertion loop
    for ins_ref_pos, ins_hyp_pos in asr_ins_errors:
        if ins_ref_pos >= len(ori_indx) or ins_ref_pos == 0:
            counts.FR += 1
            continue
        candidate_ann = ori_indx[ins_ref_pos] - 1
        if (candidate_ann >= 0
                and candidate_ann < len(pron_errors)
                and pron_errors[candidate_ann] == "d"
                and candidate_ann not in handeled_del):
            handeled_del.append(candidate_ann)
            deleted_ph  = exp_trans[candidate_ann]
            inserted_ph = (predicted[ins_hyp_pos]
                           if ins_hyp_pos < len(predicted) else "sil")
            if inserted_ph == deleted_ph:  counts.FA    += 1
            else:                          counts.TR_DE += 1
        else:
            counts.FR += 1

    # Step 4: unmatched deletion cleanup
    for ann_i, err in enumerate(pron_errors):
        if err == "d" and ann_i not in handeled_del:
            counts.TR_CD += 1

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Core counting: phonological-level (wav2vec2 SCTC-SB)
# ─────────────────────────────────────────────────────────────────────────────

def count_phonological_mdd(
    human:            list[str],
    canonical:        list[str],
    pron_errors:      list[str],
    exp_trans:        list[str],
    act_trans:        list[str],
    ori_indx:         list[int],
    predicted_logits: np.ndarray,
) -> PhonologicalMDDCounts:
    """
    Phonological feature-level MDD counting for a single utterance.

    Same four-step algorithm as count_phoneme_mdd, applied independently
    for each of the 35 features.

    For each feature f:
      1. CTC-collapse logits → binary sequence (U_f values, 0 or 1).
      2. Levenshtein-align U_f against HUMAN feature binary sequence (H values).
      3. Classify using error type and alignment result.
      4. Unmatched deletion cleanup.

    canon_matches_actual_f = (feature value of exp_trans[ori_pos]
                               == feature value of act_trans[ori_pos])
    """
    from alignment import levenshtein_alignment

    phon_counts = PhonologicalMDDCounts()
    H = len(human)
    if H == 0:
        return phon_counts

    pred_feature_seqs = _decode_sctcSB_logits_to_feature_sequences(predicted_logits)
    human_feats = np.stack([_phoneme_to_binary_array(ph) for ph in human])   # (H, 35)

    n_ann = len(pron_errors)
    canon_feats_by_ori  = np.zeros((n_ann, NUM_FEATURES), dtype=np.int8)
    actual_feats_by_ori = np.zeros((n_ann, NUM_FEATURES), dtype=np.int8)
    for ann_i in range(n_ann):
        canon_feats_by_ori[ann_i]  = _phoneme_to_binary_array(exp_trans[ann_i])
        actual_feats_by_ori[ann_i] = _phoneme_to_binary_array(act_trans[ann_i])

    for f_idx, feat_name in enumerate(PHONOLOGICAL_FEATURES):
        cnt = phon_counts.counts[feat_name]

        human_binary = human_feats[:, f_idx].astype(int).tolist()
        pred_binary  = pred_feature_seqs[f_idx]

        # Step 1: align predicted against HUMAN feature sequence
        _, _, _, ops = levenshtein_alignment(human_binary, pred_binary)

        asr_evl     = ["hit"] * H
        hyp_pos_arr = list(range(H))
        asr_ins_errors_f: list[tuple[int, int]] = []
        hyp_offset  = 0
        ref_cursor  = 0

        for op, _ref, _hyp in ops:
            if op == "I":
                asr_ins_errors_f.append((ref_cursor, ref_cursor + hyp_offset))
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

        # Step 2: main classification loop
        handeled_del_f: list[int] = []

        for ref_pos, asr_pos, ori_pos in zip(range(H), hyp_pos_arr, ori_indx):
            evl   = asr_evl[ref_pos]
            error = pron_errors[ori_pos]
            cf    = int(canon_feats_by_ori[ori_pos, f_idx])
            af    = int(actual_feats_by_ori[ori_pos, f_idx])
            cma   = (cf == af)   # canon_matches_actual for this feature

            if evl == "hit":
                if error == "c":                     cnt.TA    += 1
                elif error == "s" and cma:           cnt.TA    += 1
                elif error == "s" and not cma:       cnt.TR_CD += 1
                elif error == "a":                   cnt.TR_CD += 1

            elif evl == "replace":
                if error == "c":                     cnt.FR    += 1
                elif error == "s" and cma:           cnt.FR    += 1
                elif error == "s" and not cma:       cnt.FA    += 1
                elif error == "a":                   cnt.TR_DE += 1

            elif evl == "delete":
                if error == "c":                     cnt.FR    += 1
                elif error == "s" and cma:           cnt.FR    += 1
                elif error == "s" and not cma:       cnt.TR_DE += 1
                elif error == "a":                   cnt.FA    += 1

        # Step 3: model insertion loop
        for ins_ref_pos, ins_hyp_pos in asr_ins_errors_f:
            if ins_ref_pos >= len(ori_indx) or ins_ref_pos == 0:
                cnt.FR += 1
                continue
            candidate_ann = ori_indx[ins_ref_pos] - 1
            if (candidate_ann >= 0
                    and candidate_ann < len(pron_errors)
                    and pron_errors[candidate_ann] == "d"
                    and candidate_ann not in handeled_del_f):
                handeled_del_f.append(candidate_ann)
                deleted_f  = int(canon_feats_by_ori[candidate_ann, f_idx])
                inserted_f = (pred_binary[ins_hyp_pos]
                              if ins_hyp_pos < len(pred_binary) else -1)
                if inserted_f == deleted_f:  cnt.FA    += 1
                else:                        cnt.TR_DE += 1
            else:
                cnt.FR += 1

        # Step 4: unmatched deletion cleanup
        for ann_i, err in enumerate(pron_errors):
            if err == "d" and ann_i not in handeled_del_f:
                cnt.TR_CD += 1

    return phon_counts

# ─────────────────────────────────────────────────────────────────────────────
# Core counting: phonological-level (wav2vec2 SCTC-SB)
# ─────────────────────────────────────────────────────────────────────────────

def _phoneme_to_binary_array(phoneme: str) -> np.ndarray:
    """Return a binary feature vector of length 35 for a phoneme."""
    feat_dict = PHONEME_FEATURES.get(phoneme, PHONEME_FEATURES["sil"])
    return np.array(
        [1 if feat_dict[f] else 0 for f in PHONOLOGICAL_FEATURES],
        dtype=np.int8,
    )


def _ctc_collapse_local_sequence(local_ids: list[int], blank_id: int = 2) -> list[int]:
    """
    Collapse one CTC path: remove blank labels and repeated non-blank labels.

    For SCTC-SB each feature has a local alphabet:
        0 = +att, 1 = -att, 2 = shared blank
    """
    collapsed: list[int] = []
    prev = None
    for idx in local_ids:
        if idx == blank_id:
            prev = None
            continue
        if idx != prev:
            collapsed.append(idx)
            prev = idx
    return collapsed


def _decode_sctcSB_logits_to_feature_sequences(logits: np.ndarray) -> list[list[int]]:
    """
    Decode raw SCTC-SB logits into 35 CTC-collapsed binary feature sequences.

    This follows the paper's phonological-level idea more closely than reducing
    acoustic frames into equal-width phoneme slots: each feature category is
    decoded as its own +att/-att sequence with the shared blank removed.

    Args:
        logits: (T, 71) raw model output. Also accepts (T, 35) already-binary
                frame scores as a fallback, but true SCTC-SB evaluation should
                use the 71-node output.

    Returns:
        feature_sequences: list[35][U_f], values are 1 (+att) or 0 (-att).
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2-D logits, got shape {logits.shape}")

    decoded: list[list[int]] = []

    if logits.shape[1] == NUM_OUTPUT_NODES:
        for f in range(NUM_FEATURES):
            pos = feature_idx_to_pos_node(f)
            neg = feature_idx_to_neg_node(f)
            cat_logits = logits[:, [pos, neg, BLANK_IDX]]  # local 0,1,2
            local_path = np.argmax(cat_logits, axis=-1).astype(int).tolist()
            collapsed = _ctc_collapse_local_sequence(local_path, blank_id=2)
            decoded.append([1 if x == 0 else 0 for x in collapsed])
        return decoded

    if logits.shape[1] == NUM_FEATURES:
        # Fallback for already-compressed output. This is not full SCTC-SB
        # decoding because no blank node is present, but it keeps compatibility.
        binary_frames = (logits > 0).astype(np.int8)
        for f in range(NUM_FEATURES):
            seq = binary_frames[:, f].astype(int).tolist()
            collapsed = []
            prev = None
            for x in seq:
                if x != prev:
                    collapsed.append(x)
                    prev = x
            decoded.append(collapsed)
        return decoded

    raise ValueError(
        f"Unexpected logits last dim {logits.shape[1]}; "
        f"expected {NUM_OUTPUT_NODES} (71-node output) or {NUM_FEATURES}."
    )


class MDDEvaluator:
    """
    Accumulates MDD counts across utterances and computes final metrics.

    Supports phoneme-level (Whisper) and/or phonological-level (SCTC-SB)
    evaluation. Both can run simultaneously on the same utterances.

    Typical usage
    -------------
        evaluator = MDDEvaluator()

        for ann_file in annotation_dir.glob("*.TextGrid"):
            (human, canonical, pron_errors,
             exp_trans, act_trans, ori_indx) = parse_annotation_for_mdd(str(ann_file))
            utt_id = ann_file.stem

            # Phoneme-level
            predicted_phones = _run_phoneme_wav2vec2(phoneme_model, feat_extractor, waveform, device)
            evaluator.add_phoneme_utterance(
                human, canonical, pron_errors, exp_trans, act_trans, ori_indx,
                predicted_phones, utt_id,
            )

            # Phonological-level (wav2vec2 SCTC-SB)
            logits = _run_phonological_wav2vec2(phonological_model, feat_extractor, waveform, device)
            evaluator.add_phonological_utterance(
                human, canonical, pron_errors, exp_trans, act_trans, ori_indx,
                logits, utt_id,
            )

        evaluator.print_report()
        evaluator.save_json("mdd_results.json")
    """

    def __init__(self):
        self.phoneme_counts      = MDDCounts()
        self.phonological_counts = PhonologicalMDDCounts()
        self.n_phoneme_utts      = 0
        self.n_phonological_utts = 0

    def add_phoneme_utterance(
        self,
        human:       list[str],
        canonical:   list[str],
        pron_errors: list[str],
        exp_trans:   list[str],
        act_trans:   list[str],
        ori_indx:    list[int],
        predicted:   list[str],
        utt_id:      Optional[str] = None,
    ) -> MDDCounts:
        """Add one utterance to the phoneme-level accumulator."""
        if not human:
            logger.warning(f"[{utt_id}] empty human sequence, skipping")
            return MDDCounts()
        utt_counts = count_phoneme_mdd(
            human, canonical, pron_errors, exp_trans, act_trans, ori_indx, predicted
        )
        self.phoneme_counts = self.phoneme_counts + utt_counts
        self.n_phoneme_utts += 1
        return utt_counts

    def add_phonological_utterance(
        self,
        human:            list[str],
        canonical:        list[str],
        pron_errors:      list[str],
        exp_trans:        list[str],
        act_trans:        list[str],
        ori_indx:         list[int],
        predicted_logits: np.ndarray,
        utt_id:           Optional[str] = None,
    ) -> PhonologicalMDDCounts:
        """Add one utterance to the phonological-level (SCTC-SB) accumulator."""
        if not human:
            logger.warning(f"[{utt_id}] empty human sequence, skipping")
            return PhonologicalMDDCounts()
        if predicted_logits.ndim != 2 or predicted_logits.shape[1] not in (
            NUM_FEATURES, NUM_OUTPUT_NODES
        ):
            raise ValueError(
                f"[{utt_id}] predicted_logits must be (T, {NUM_FEATURES}) "
                f"or (T, {NUM_OUTPUT_NODES}), got {predicted_logits.shape}"
            )
        utt_counts = count_phonological_mdd(
            human, canonical, pron_errors, exp_trans, act_trans, ori_indx, predicted_logits
        )
        self.phonological_counts.add(utt_counts)
        self.n_phonological_utts += 1
        return utt_counts

    def compute(self) -> dict:
        """Return full results dict."""
        return {
            "phoneme_level":      self.phoneme_counts.summary(),
            "phonological_level": self.phonological_counts.summary(),
            "counts": {
                "n_phoneme_utts":      self.n_phoneme_utts,
                "n_phonological_utts": self.n_phonological_utts,
            },
        }

    def print_report(self):
        results = self.compute()
        print("=" * 70)
        print("  MDD EVALUATION RESULTS")
        print("=" * 70)

        if self.n_phoneme_utts > 0:
            p = results["phoneme_level"]
            print(f"\n{'─'*70}")
            print(f"  PHONEME-LEVEL  (wav2vec2 CTC)  --  {self.n_phoneme_utts} utterances")
            print(f"{'─'*70}")
            print(f"  TA={p['TA']}  FA={p['FA']}  FR={p['FR']}  "
                  f"TR={p['TR']}  (CD={p['TR_CD']}, DE={p['TR_DE']})")
            print(f"  FAR={p['FAR']:.4f}   FRR={p['FRR']:.4f}   DER={p['DER']:.4f}")

        if self.n_phonological_utts > 0:
            phon  = dict(results["phonological_level"])
            macro = phon.pop("__macro_avg__")
            print(f"\n{'─'*70}")
            print(f"  PHONOLOGICAL-LEVEL  (SCTC-SB)  --  "
                  f"{self.n_phonological_utts} utterances")
            print(f"{'─'*70}")
            print(f"  {'Feature':<20}  {'TA':>6}  {'FA':>6}  {'FR':>6}  "
                  f"{'CD':>6}  {'DE':>6}  {'FAR':>7}  {'FRR':>7}  {'DER':>7}")
            print(f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*6}  "
                  f"{'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
            for feat, m in phon.items():
                print(
                    f"  {feat:<20}  {m['TA']:>6}  {m['FA']:>6}  {m['FR']:>6}  "
                    f"{m['TR_CD']:>6}  {m['TR_DE']:>6}  "
                    f"{m['FAR']:>7.4f}  {m['FRR']:>7.4f}  {m['DER']:>7.4f}"
                )
            print(f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*6}  "
                  f"{'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
            print(
                f"  {'MACRO AVG':<20}  {'':>6}  {'':>6}  {'':>6}  "
                f"{'':>6}  {'':>6}  "
                f"{macro['FAR']:>7.4f}  {macro['FRR']:>7.4f}  {macro['DER']:>7.4f}"
            )

        print("=" * 70)

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.compute(), f, indent=2)
        print(f"[MDDEvaluator] Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model inference helpers
# ─────────────────────────────────────────────────────────────────────────────

# Phoneme index → string mapping for PhonemeLevelWav2Vec2 CTC decoding.
# Imported directly from phonological_features so this always stays in sync
# with the label vocabulary used during training.
# CMU_39_PHONEMES has 39 entries (indices 0-38); index 39 = blank.
# "sil" is the last entry (index 38) and is filtered out after decoding.
from phonological_features import CMU_39_PHONEMES as _PHONEME_IDX_TO_STR


def _run_phoneme_wav2vec2(
    model,
    feature_extractor,
    waveform,
    device: str,
    blank_idx: int = 39,
) -> list[str]:
    """
    Run PhonemeLevelWav2Vec2 → predicted phoneme sequence.

    Greedy CTC decoding on the (T, 40) logits:
      1. argmax over vocab at each frame
      2. collapse consecutive repeated tokens
      3. remove blank (index 39)
      4. map remaining indices to CMU-39 phoneme strings

    Args:
        model             : PhonemeLevelWav2Vec2 (eval mode, on device)
        feature_extractor : Wav2Vec2FeatureExtractor
        waveform          : 1-D float tensor, 16 kHz mono
        device            : "cuda" or "cpu"
        blank_idx         : CTC blank index (default 39)

    Returns:
        List of normalised CMU-39 phoneme strings, silences removed.
    """
    import torch
    inputs = feature_extractor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits, _ = model(inputs.input_values.to(device))  # (1, T, 40)

    logit_ids = logits.squeeze(0).argmax(dim=-1).tolist()  # greedy: list[int]

    # CTC collapse: remove blanks and consecutive repeats
    phones: list[str] = []
    prev = None
    for idx in logit_ids:
        if idx == blank_idx:
            prev = None
            continue
        if idx != prev:
            ph = _PHONEME_IDX_TO_STR[idx] if idx < len(_PHONEME_IDX_TO_STR) else "sil"
            if ph != "sil":
                phones.append(ph)
            prev = idx

    return phones


def _run_phonological_wav2vec2(
    model,
    feature_extractor,
    waveform,
    device: str,
) -> np.ndarray:
    """
    Run PhonologicalWav2Vec2 (SCTC-SB) → raw logits (T, 71) as a numpy array.

    Logits are trimmed to the valid (non-padded) frame count using the
    output_lengths returned by the model's forward pass.
    """
    import torch
    inputs = feature_extractor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = model(inputs.input_values.to(device),
                        inputs.get("attention_mask", None))
    if isinstance(outputs, (tuple, list)):
        logits, output_lengths = outputs[0], outputs[1]
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
        output_lengths = None
    else:
        logits = outputs
        output_lengths = None

    logits_np = logits.squeeze(0).cpu().numpy()   # (T, 71)

    # Trim to valid frames — removes any padding frames at the tail
    if output_lengths is not None:
        valid_len = int(output_lengths[0].item())
        logits_np = logits_np[:valid_len]
    
    return logits_np


# ─────────────────────────────────────────────────────────────────────────────
# Suitcase TextGrid parser — timestamp-aware, returns MDD structures per chunk
# ─────────────────────────────────────────────────────────────────────────────

def _parse_suitcase_textgrid_for_mdd(textgrid_path: str) -> list[dict]:
    """
    Parse a suitcase annotation TextGrid and return one record per interval
    with timestamps and full MDD annotation fields.

    Each record:
        xmin, xmax   : float  — interval boundaries in seconds
        is_silence   : bool
        error        : str    — 'c', 's', 'd', 'a'   (None for silence)
        exp_phone    : str    — canonical phoneme      (None for silence)
        act_phone    : str    — actual phoneme (None if deleted or silence)

    These records are later chunked and converted to the five sequences
    expected by count_phoneme_mdd / count_phonological_mdd.
    """
    import re as _re
    path = Path(textgrid_path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    tier_blocks = _re.split(r'item\s*\[\d+\]', content)
    phones_block = None
    for block in tier_blocks:
        if _re.search(r'name\s*=\s*"phones?"', block, re.IGNORECASE):
            phones_block = block
            break
    if phones_block is None:
        return []

    intervals = _re.findall(
        r'intervals\s*\[\d+\].*?xmin\s*=\s*([\d.]+).*?xmax\s*=\s*([\d.]+)'
        r'.*?text\s*=\s*"([^"]*)"',
        phones_block, re.DOTALL,
    )

    records = []
    for xmin_s, xmax_s, text in intervals:
        xmin, xmax = float(xmin_s), float(xmax_s)
        text = text.strip()

        if text in ("", "sil", "sp", "spn", "<unk>"):
            records.append({"xmin": xmin, "xmax": xmax, "is_silence": True,
                            "error": None, "exp_phone": None, "act_phone": None})
            continue

        parts = [p.strip() for p in text.split(",")]

        if len(parts) == 1:
            ph = normalize_phoneme(parts[0])
            if ph == "sil":
                records.append({"xmin": xmin, "xmax": xmax, "is_silence": True,
                                "error": None, "exp_phone": None, "act_phone": None})
            else:
                records.append({"xmin": xmin, "xmax": xmax, "is_silence": False,
                                "error": "c", "exp_phone": ph, "act_phone": ph})

        elif len(parts) == 3:
            canonical_raw, pronounced_raw, error_type = parts
            error_type = error_type.strip().lower()

            if error_type == "s":
                canon_ph = normalize_phoneme(canonical_raw)
                if canon_ph == "sil":
                    records.append({"xmin": xmin, "xmax": xmax, "is_silence": True,
                                    "error": None, "exp_phone": None, "act_phone": None})
                    continue
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                act_ph = ("sil" if pronounced_clean.lower() == "err"
                          else normalize_phoneme(pronounced_clean))
                records.append({"xmin": xmin, "xmax": xmax, "is_silence": False,
                                "error": "s", "exp_phone": canon_ph,
                                "act_phone": act_ph if act_ph != "sil" else None})

            elif error_type == "d":
                canon_ph = normalize_phoneme(canonical_raw)
                if canon_ph == "sil":
                    records.append({"xmin": xmin, "xmax": xmax, "is_silence": True,
                                    "error": None, "exp_phone": None, "act_phone": None})
                    continue
                # Deletion: no audio produced, but keep for MDD step 3/4
                records.append({"xmin": xmin, "xmax": xmax, "is_silence": False,
                                "error": "d", "exp_phone": canon_ph, "act_phone": None})

            elif error_type == "a":
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                act_ph = normalize_phoneme(pronounced_clean)
                if act_ph == "sil":
                    records.append({"xmin": xmin, "xmax": xmax, "is_silence": True,
                                    "error": None, "exp_phone": None, "act_phone": None})
                    continue
                records.append({"xmin": xmin, "xmax": xmax, "is_silence": False,
                                "error": "a", "exp_phone": "sil", "act_phone": act_ph})

    return records


def _chunk_suitcase_records(
    records: list[dict],
    max_chunk_duration: float = 10.0,
) -> list[list[dict]]:
    """
    Split suitcase interval records into chunks ≤ max_chunk_duration seconds,
    preferring silence boundaries (mirrors dataset.py _chunk_records).
    """
    if not records:
        return []
    chunks, current, chunk_start = [], [], records[0]["xmin"]
    for rec in records:
        duration = rec["xmax"] - chunk_start
        if rec["is_silence"] and duration >= max_chunk_duration * 0.5:
            if current:
                chunks.append(current)
            current, chunk_start = [], rec["xmax"]
            continue
        if duration >= max_chunk_duration:
            if current:
                chunks.append(current)
            current, chunk_start = [rec], rec["xmin"]
        else:
            current.append(rec)
    if current:
        chunks.append(current)
    return chunks


def _suitcase_chunk_to_mdd_sequences(
    chunk: list[dict],
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[int]]:
    """
    Convert a list of suitcase interval records (one chunk) into the five
    sequences expected by count_phoneme_mdd / count_phonological_mdd.

    Mirrors parse_annotation_for_mdd() exactly — deletions stay in
    pron_errors but are absent from human; additions are in human.
    """
    pron_errors: list[str] = []
    exp_trans:   list[str] = []
    act_trans:   list[str] = []
    human:       list[str] = []
    ori_indx:    list[int] = []
    ann_idx = 0

    for rec in chunk:
        if rec["is_silence"] or rec["error"] is None:
            continue

        error     = rec["error"]
        exp_phone = rec["exp_phone"] or "sil"
        act_phone = rec["act_phone"] or "sil"

        if error == "c":
            pron_errors.append("c")
            exp_trans.append(exp_phone)
            act_trans.append(act_phone)
            human.append(act_phone)
            ori_indx.append(ann_idx)
            ann_idx += 1

        elif error == "s":
            pron_errors.append("s")
            exp_trans.append(exp_phone)
            act_trans.append(act_phone)
            if act_phone != "sil":
                human.append(act_phone)
                ori_indx.append(ann_idx)
            ann_idx += 1

        elif error == "d":
            # Deletion — not in human, kept for two-pass cleanup
            pron_errors.append("d")
            exp_trans.append(exp_phone)
            act_trans.append("sil")
            ann_idx += 1

        elif error == "a":
            pron_errors.append("a")
            exp_trans.append("sil")
            act_trans.append(act_phone)
            human.append(act_phone)
            ori_indx.append(ann_idx)
            ann_idx += 1

    canonical = [exp_trans[ori_indx[i]] for i in range(len(human))]
    return human, canonical, pron_errors, exp_trans, act_trans, ori_indx


# ─────────────────────────────────────────────────────────────────────────────
# Suitcase corpus evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mdd_suitcase(
    l2arctic_root:        str,
    speakers:             list[str],
    phoneme_model         = None,
    phonological_model    = None,
    feature_extractor     = None,
    device:      str        = "cuda",
    output_json: Optional[str] = None,
    max_chunk_duration: float = 10.0,
    verbose:     bool       = True,
) -> dict:
    """
    Run MDD evaluation on the L2-ARCTIC suitcase (spontaneous speech) corpus.

    Each speaker has ONE long WAV + ONE fully-annotated TextGrid under:
        <l2arctic_root>/suitcase_corpus/wav/<spk_lower>.wav
        <l2arctic_root>/suitcase_corpus/annotation/<spk_lower>.TextGrid

    The long recording is split into chunks of ≤ max_chunk_duration seconds
    using silence boundaries from the TextGrid timestamps.  Each chunk is
    treated as one utterance for evaluation purposes.

    Args:
        l2arctic_root      : root path of the L2-ARCTIC corpus
        speakers           : list of speaker IDs (e.g. ["RRBI", "YBAA", ...])
        phoneme_model      : PhonemeLevelWav2Vec2 instance (optional)
        phonological_model : PhonologicalWav2Vec2 instance (optional)
        feature_extractor  : Wav2Vec2FeatureExtractor (shared by both models)
        device             : "cuda" or "cpu"
        output_json        : if set, save results to this path as JSON
        max_chunk_duration : max chunk length in seconds (default 10.0)
        verbose            : log progress

    Returns:
        dict with keys: phoneme_level, phonological_level, counts
    """
    import torch
    import torchaudio

    evaluator = MDDEvaluator()
    suit_root = Path(l2arctic_root) / "suitcase_corpus"
    wav_dir   = suit_root / "wav"
    ann_dir   = suit_root / "annotation"

    if phoneme_model is not None:
        phoneme_model.to(device).eval()
    if phonological_model is not None:
        phonological_model.to(device).eval()

    n_total = n_skipped = 0

    for spk in speakers:
        # Suitcase files are named with lowercase speaker ID
        wav_file = wav_dir / f"{spk.lower()}.wav"
        tg_file  = ann_dir / f"{spk.lower()}.TextGrid"

        if not wav_file.exists():
            logger.warning(f"[suitcase] wav not found for {spk}: {wav_file}")
            continue
        if not tg_file.exists():
            logger.warning(f"[suitcase] TextGrid not found for {spk}: {tg_file}")
            continue

        # Parse the full TextGrid into MDD-annotated interval records
        records = _parse_suitcase_textgrid_for_mdd(str(tg_file))
        if not records:
            logger.warning(f"[suitcase] no records parsed for {spk}, skipping")
            continue

        # Load native sample rate (don't load the full audio yet)
        _, native_sr = torchaudio.load(str(wav_file), num_frames=1)

        chunks = _chunk_suitcase_records(records, max_chunk_duration)
        if verbose:
            print(f"  [{spk}] suitcase: {len(chunks)} chunks from {tg_file.name}")

        for chunk_idx, chunk in enumerate(chunks):
            n_total += 1
            utt_id = f"{spk.lower()}_{chunk_idx:03d}"

            if verbose and chunk_idx % 50 == 0 and chunk_idx > 0:
                print(f"    [{spk}] chunk {chunk_idx}/{len(chunks)}")

            # Build MDD sequences from this chunk
            (human, canonical, pron_errors,
             exp_trans, act_trans, ori_indx) = _suitcase_chunk_to_mdd_sequences(chunk)

            if not human:
                n_skipped += 1
                continue

            # Slice audio for this chunk (timestamp → sample index)
            start_sample = int(chunk[0]["xmin"] * native_sr)
            end_sample   = int(chunk[-1]["xmax"] * native_sr)
            num_frames   = end_sample - start_sample
            if num_frames <= 0:
                n_skipped += 1
                continue

            waveform, sr = torchaudio.load(
                str(wav_file),
                frame_offset=start_sample,
                num_frames=num_frames,
            )
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.mean(dim=0)   # mono (T,)

            # Phoneme-level MDD
            if phoneme_model is not None and feature_extractor is not None:
                try:
                    predicted_phones = _run_phoneme_wav2vec2(
                        phoneme_model, feature_extractor, waveform, device
                    )
                    evaluator.add_phoneme_utterance(
                        human, canonical, pron_errors, exp_trans,
                        act_trans, ori_indx, predicted_phones, utt_id
                    )
                except Exception as e:
                    logger.warning(f"[{utt_id}] phoneme model inference error: {e}")

            # Phonological-level MDD
            if phonological_model is not None and feature_extractor is not None:
                try:
                    logits = _run_phonological_wav2vec2(
                        phonological_model, feature_extractor, waveform, device
                    )
                    evaluator.add_phonological_utterance(
                        human, canonical, pron_errors, exp_trans,
                        act_trans, ori_indx, logits, utt_id
                    )
                except Exception as e:
                    logger.warning(f"[{utt_id}] phonological model inference error: {e}")

    print(f"\n[evaluate_mdd_suitcase] total={n_total}  skipped={n_skipped}  "
          f"evaluated={n_total - n_skipped}")
    evaluator.print_report()

    if output_json:
        evaluator.save_json(output_json)

    return evaluator.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Full corpus evaluation loop

def evaluate_mdd(
    l2arctic_root:        str,
    speakers:             list[str],
    phoneme_model         = None,
    phonological_model    = None,
    feature_extractor     = None,
    device:      str        = "cuda",
    output_json: Optional[str] = None,
    verbose:     bool       = True,
) -> dict:
    """
    Run full MDD evaluation over L2-ARCTIC annotated utterances.

    Only utterances that have a corresponding annotation/<utt_id>.TextGrid are
    evaluated. Unannotated utterances are skipped — there is no fallback.

    Both canonical and human-annotated sequences are derived directly from the
    annotation TextGrid via parse_annotation_for_mdd().

    Directory structure assumed per speaker:
        <l2arctic_root>/<speaker>/
          wav/          ← <utt_id>.wav  (audio files)
          annotation/   ← <utt_id>.TextGrid  (human MDD annotations, ~150/speaker)

    Args:
        l2arctic_root      : root path of the L2-ARCTIC corpus
        speakers           : list of speaker IDs to evaluate
        phoneme_model      : PhonemeLevelWav2Vec2 instance (optional)
        phonological_model : PhonologicalWav2Vec2 instance (optional)
        feature_extractor  : Wav2Vec2FeatureExtractor (shared by both models)
        device             : "cuda" or "cpu"
        output_json        : if set, save results to this path as JSON
        verbose            : log progress every 50 utterances

    Returns:
        dict with keys: phoneme_level, phonological_level, counts
    """
    import torch
    import torchaudio

    evaluator = MDDEvaluator()
    root      = Path(l2arctic_root)

    if phoneme_model is not None:
        phoneme_model.to(device).eval()
    if phonological_model is not None:
        phonological_model.to(device).eval()

    n_total = n_skipped = 0

    for spk in speakers:
        spk_dir  = root / spk
        wav_dir  = spk_dir / "wav"
        ann_dir  = spk_dir / "annotation"

        if not wav_dir.exists():
            logger.warning(f"wav dir not found for speaker {spk}, skipping")
            continue
        if not ann_dir.exists():
            logger.warning(f"annotation dir not found for speaker {spk}, skipping")
            continue

        # Drive the loop from annotation files — only annotated utterances
        ann_files = sorted(ann_dir.glob("*.TextGrid"))
        if not ann_files:
            logger.warning(f"No annotation TextGrids found for speaker {spk}, skipping")
            continue

        for ann_idx, ann_file in enumerate(ann_files):
            utt_id   = ann_file.stem
            wav_file = wav_dir / f"{utt_id}.wav"
            n_total += 1

            if verbose and ann_idx % 50 == 0:
                print(f"  [{spk}] {ann_idx + 1}/{len(ann_files)}  {utt_id}")

            # ── 1. Canonical + Human: both from annotation TextGrid ───────
            (human, canonical, pron_errors,
             exp_trans, act_trans, ori_indx) = parse_annotation_for_mdd(str(ann_file))
            if not human:
                logger.warning(f"[{utt_id}] no human phones from annotation, skipping")
                n_skipped += 1
                continue

            # ── 2. Load audio ─────────────────────────────────────────────
            if not wav_file.exists():
                logger.warning(f"[{utt_id}] wav file not found: {wav_file}, skipping")
                n_skipped += 1
                continue

            waveform, sr = torchaudio.load(str(wav_file))
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.mean(dim=0)   # mono (T,)

            # ── 3. Phoneme-level MDD (PhonemeLevelWav2Vec2 CTC) ──────────
            if phoneme_model is not None and feature_extractor is not None:
                try:
                    predicted_phones = _run_phoneme_wav2vec2(
                        phoneme_model, feature_extractor, waveform, device
                    )
                    evaluator.add_phoneme_utterance(
                        human, canonical, pron_errors, exp_trans,
                        act_trans, ori_indx, predicted_phones, utt_id
                    )
                except Exception as e:
                    logger.warning(f"[{utt_id}] phoneme model inference error: {e}")

            # ── 4. Phonological-level MDD (PhonologicalWav2Vec2 SCTC-SB) ─
            if phonological_model is not None and feature_extractor is not None:
                try:
                    logits = _run_phonological_wav2vec2(
                        phonological_model, feature_extractor, waveform, device
                    )
                    evaluator.add_phonological_utterance(
                        human, canonical, pron_errors, exp_trans,
                        act_trans, ori_indx, logits, utt_id
                    )
                except Exception as e:
                    logger.warning(f"[{utt_id}] phonological model inference error: {e}")

    print(f"\n[evaluate_mdd] total={n_total}  skipped={n_skipped}  "
          f"evaluated={n_total - n_skipped}")
    evaluator.print_report()

    if output_json:
        evaluator.save_json(output_json)

    return evaluator.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check — reproduces Table 2 from Shahin et al. (2025)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="MDD evaluation for phoneme-level (PhonemeLevelWav2Vec2 CTC) "
                    "and/or phonological-level (PhonologicalWav2Vec2 SCTC-SB) models."
    )
    parser.add_argument("--l2arctic_dir",        type=str, default=None)
    parser.add_argument("--phoneme_model",        type=str, default=None,
                        help="Path to PhonemeLevelWav2Vec2 checkpoint (.pt)")
    parser.add_argument("--phonological_model",   type=str, default=None,
                        help="Path to PhonologicalWav2Vec2 checkpoint (.pt)")
    parser.add_argument("--feature_extractor",    type=str,
                        default="facebook/wav2vec2-large-robust",
                        help="HuggingFace model name for Wav2Vec2FeatureExtractor")
    parser.add_argument("--speakers",             type=str, nargs="+", default=None)
    parser.add_argument("--output_json",          type=str, default=None)
    parser.add_argument("--device",               type=str, default="cuda")
    parser.add_argument("--sanity_check",         action="store_true")
    parser.add_argument("--suitcase",             action="store_true",
                        help="Also run MDD evaluation on the suitcase corpus "
                             "(l2arctic_dir/suitcase_corpus/). Results are saved "
                             "separately to <output_json>.suitcase.json if "
                             "--output_json is given.")
    parser.add_argument("--suitcase_only",        action="store_true",
                        help="Run ONLY the suitcase evaluation (skip scripted).")
    parser.add_argument("--suitcase_speakers",    type=str, nargs="+", default=None,
                        help="Suitcase speaker IDs to evaluate. Defaults to the "
                             "same list as --speakers (or DEFAULT_TEST_SPEAKERS).")
    parser.add_argument("--max_chunk_duration",   type=float, default=10.0,
                        help="Max chunk duration in seconds for suitcase chunking "
                             "(default: 10.0).")
    args = parser.parse_args()


    # ── Sanity check test data ────────────────────────────────────────────────
    # Simulates one utterance with 5 canonical positions:
    #   pos 0: ae  correct (human=ae)        → should be TA
    #   pos 1: d   substitution (human=t)    → FA if model predicts d, TR if not
    #   pos 2: v   correct (human=v)         → TA if model predicts v, FR if not
    #   pos 3: ey  substitution (human=ay)   → FA if model predicts ey
    #   pos 4: s   deletion (human=None)     → FR if model predicts s, TR if not
    # Sanity-check utterance: "tab" with one substitution, one deletion, one addition
    #   ann idx 0: ae  error=c  exp=ae  act=ae   (correct)
    #   ann idx 1: d   error=s  exp=d   act=t    (substitution: d→t)
    #   ann idx 2: v   error=c  exp=v   act=v    (correct)
    #   ann idx 3: ey  error=s  exp=ey  act=ay   (substitution: ey→ay)
    #   ann idx 4: s   error=d  exp=s   act=sil  (deletion: speaker omitted s)
    #   ann idx 5: m   error=a  exp=sil act=m    (addition: speaker added m)
    #
    # human    = phones speaker actually said (no deletions, additions included)
    # ori_indx = maps human position → ann idx
    _pron_errors = ["c",   "s",  "c",  "s",   "d",   "a"]
    _exp_trans   = ["ae",  "d",  "v",  "ey",  "s",   "sil"]
    _act_trans   = ["ae",  "t",  "v",  "ay",  "sil", "m"]
    _human       = ["ae",  "t",  "v",  "ay",  "m"]   # no deletion, addition included
    _ori_indx    = [0,     1,    2,    3,     5]      # maps to ann indices
    _canonical   = [_exp_trans[i] for i in _ori_indx] # ["ae","d","v","ey","sil"]

    # predicted = canonical (model output matches expected, not actual)
    #   pos 0: human=ae  pred=ae  error=c  → hit+c → TA
    #   pos 1: human=t   pred=d   error=s  exp=d act=t exp≠act → replace+s(exp≠act) → FA
    #   pos 2: human=v   pred=v   error=c  → hit+c → TA
    #   pos 3: human=ay  pred=ey  error=s  exp=ey act=ay exp≠act → replace+s(exp≠act) → FA
    #   pos 4: human=m   pred=sil error=a  exp=sil → replace+a → TR_DE
    #   ann idx 4 (d error) unmatched → TR_CD
    _predicted_canonical = ["ae", "d", "v", "ey", "sil"]

    print("  Testing count_phoneme_mdd with predicted=canonical ...")
    ph_result = count_phoneme_mdd(
        _human, _canonical, _pron_errors, _exp_trans, _act_trans, _ori_indx,
        _predicted_canonical,
    )
    assert ph_result.TA    == 2, f"Expected TA=2,    got {ph_result.TA}"
    assert ph_result.FA    == 2, f"Expected FA=2,    got {ph_result.FA}"
    assert ph_result.TR_DE == 1, f"Expected TR_DE=1, got {ph_result.TR_DE}"
    assert ph_result.TR_CD == 1, f"Expected TR_CD=1, got {ph_result.TR_CD}"
    assert ph_result.FR    == 0, f"Expected FR=0,    got {ph_result.FR}"
    print(f"    TA={ph_result.TA} FA={ph_result.FA} FR={ph_result.FR} "
          f"TR_CD={ph_result.TR_CD} TR_DE={ph_result.TR_DE}  PASSED")

    if args.sanity_check:
        sys.exit(0)

    # ── Full corpus evaluation ────────────────────────────────────────────────
    if args.l2arctic_dir is None:
        parser.error("--l2arctic_dir is required for corpus evaluation.")
    if args.phoneme_model is None and args.phonological_model is None:
        parser.error("Provide at least one of --phoneme_model or --phonological_model.")

    import torch
    from transformers import Wav2Vec2FeatureExtractor

    # L2-ARCTIC scripted test speakers (Ye et al. 2022 / Shahin et al. 2025)
    DEFAULT_TEST_SPEAKERS = ["RRBI", "YBAA", "HJK", "BWC", "EBVS", "YDCK"]
    speakers = args.speakers if args.speakers else DEFAULT_TEST_SPEAKERS
    # Suitcase speakers default to the same list unless overridden
    suit_speakers = args.suitcase_speakers if args.suitcase_speakers else speakers

    print(f"Device: {args.device}\n")

    feature_extractor  = Wav2Vec2FeatureExtractor.from_pretrained(args.feature_extractor)
    phoneme_model      = None
    phonological_model = None

    if args.phoneme_model:
        from wav2vec2_phonological import PhonemeLevelWav2Vec2
        print(f"[phoneme] Loading checkpoint: {args.phoneme_model}")
        phoneme_model = PhonemeLevelWav2Vec2()
        state = torch.load(args.phoneme_model, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        phoneme_model.load_state_dict(state)
        phoneme_model.eval()
        print("[phoneme] Ready.")

    if args.phonological_model:
        from wav2vec2_phonological import PhonologicalWav2Vec2
        print(f"[phonological] Loading checkpoint: {args.phonological_model}")
        phonological_model = PhonologicalWav2Vec2()
        state = torch.load(args.phonological_model, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        phonological_model.load_state_dict(state)
        phonological_model.eval()
        print("[phonological] Ready.")

    # ── Scripted evaluation ───────────────────────────────────────────────────
    if not args.suitcase_only:
        print(f"Evaluating scripted speakers: {speakers}")
        evaluate_mdd(
            l2arctic_root      = args.l2arctic_dir,
            speakers           = speakers,
            phoneme_model      = phoneme_model,
            phonological_model = phonological_model,
            feature_extractor  = feature_extractor,
            device             = args.device,
            output_json        = args.output_json,
            verbose            = True,
        )

    # ── Suitcase evaluation ───────────────────────────────────────────────────
    if args.suitcase or args.suitcase_only:
        # Derive suitcase output path from --output_json if given
        suit_json = None
        if args.output_json:
            base = args.output_json
            if base.endswith(".json"):
                suit_json = base[:-5] + "_suitcase.json"
            else:
                suit_json = base + "_suitcase.json"

        print(f"\nEvaluating suitcase speakers: {suit_speakers}")
        evaluate_mdd_suitcase(
            l2arctic_root      = args.l2arctic_dir,
            speakers           = suit_speakers,
            phoneme_model      = phoneme_model,
            phonological_model = phonological_model,
            feature_extractor  = feature_extractor,
            device             = args.device,
            output_json        = suit_json,
            max_chunk_duration = args.max_chunk_duration,
            verbose            = True,
        )
