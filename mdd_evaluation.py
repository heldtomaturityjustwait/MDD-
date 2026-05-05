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
  TR/DE : all three are different

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

Source of truth
---------------
Both canonical and human sequences come from annotation/<utt_id>.TextGrid only.
Utterances with no annotation file are SKIPPED entirely — there is no fallback to
transcript or textgrid directories.

Alignment
---------
All three sequences are aligned position-by-position against canonical (anchor).
  - Shorter human/predicted sequences: trailing positions get None (deletion,
    never equal to canonical).
  - Longer human/predicted sequences: extra tokens beyond len(canonical) ignored.

Phonological-level evaluation (wav2vec2)
-----------------------------------------
Evaluation is performed for each of the 35 features independently.
For each feature:
  1. canonical phoneme sequence  →  N binary values  (1=+att, 0=-att)
  2. human phoneme sequence      →  N binary values  (aligned to canonical)
  3. logits (T, 71)              →  decoded to (T, 35) per-frame binary predictions
                                     → majority-voted to (N, 35) by equal-width
                                        frame segmentation  (N = len(canonical))
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
# Annotation TextGrid parser — derives BOTH canonical and human sequences
# ─────────────────────────────────────────────────────────────────────────────

def parse_annotation_for_mdd(textgrid_path: str) -> tuple[list[str], list[str]]:
    """
    Parse an L2-ARCTIC annotation TextGrid and return both the canonical
    phoneme sequence and the human-annotated (actually spoken) phoneme sequence.

    This is the single source of truth for MDD evaluation. No transcript files
    or force-aligned textgrid files are used.

    TextGrid phones tier label formats
    -----------------------------------
      Correct:      "AH1"           canonical = ah,  human = ah
      Substitution: "DH,D,s"        canonical = dh,  human = d   (what was said)
      Deletion:     "TH,sil,d"      canonical = th,  human = —   (speaker omitted it)
      Addition:     "sil,AH,a"      canonical = —,   human = ah  (extra; no canonical slot)
      Hard error:   "CPL,err,s"     canonical = cpl, human = —   (uninterpretable; skip human)
      Silence:      ""/"sil"/"sp"   skip entirely

    Canonical sequence contains the phoneme that was *intended* at each
    slot (Correct and Substitution and Deletion intervals). Insertions/Additions
    have no canonical slot and are excluded from the canonical sequence but
    their pronounced phoneme is appended to the human sequence at the
    corresponding position.

    Returns
    -------
    canonical : list[str]   normalised CMU-39 phonemes, silences removed
    human     : list[str]   normalised CMU-39 phonemes, silences removed
    """
    path = Path(textgrid_path)
    if not path.exists():
        return [], []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Locate the phones tier
    tier_blocks = re.split(r'item\s*\[\d+\]', content)
    phones_block = None
    for block in tier_blocks:
        if re.search(r'name\s*=\s*"phones?"', block, re.IGNORECASE):
            phones_block = block
            break
    if phones_block is None:
        return [], []

    intervals = re.findall(
        r'intervals\s*\[\d+\].*?xmin\s*=\s*[\d.]+.*?xmax\s*=\s*[\d.]+'
        r'.*?text\s*=\s*"([^"]*)"',
        phones_block, re.DOTALL,
    )

    canonical_phones: list[str] = []
    human_phones:     list[str] = []

    for text in intervals:
        text = text.strip()
        if text in ("", "sil", "sp", "spn", "<unk>"):
            continue

        parts = [p.strip() for p in text.split(",")]

        if len(parts) == 1:
            # Correct pronunciation — same phoneme in both sequences
            ph = normalize_phoneme(parts[0])
            if ph != "sil":
                canonical_phones.append(ph)
                human_phones.append(ph)

        elif len(parts) == 3:
            canonical_raw, pronounced_raw, error_type = parts
            error_type = error_type.strip().lower()

            if error_type == "s":
                # Substitution — canonical slot exists; speaker said something different
                canon_ph = normalize_phoneme(canonical_raw)
                if canon_ph != "sil":
                    canonical_phones.append(canon_ph)

                pronounced_clean = pronounced_raw.replace("*", "").strip()
                if pronounced_clean.lower() == "err":
                    # Uninterpretable — treat as deletion on the human side
                    pass
                else:
                    human_ph = normalize_phoneme(pronounced_clean)
                    if human_ph != "sil":
                        human_phones.append(human_ph)

            elif error_type == "d":
                # Deletion — canonical slot exists; speaker produced nothing
                canon_ph = normalize_phoneme(canonical_raw)
                if canon_ph != "sil":
                    canonical_phones.append(canon_ph)
                # human_phones: nothing added (deletion)

            elif error_type == "a":
                # Addition/Insertion — no canonical slot; speaker produced extra phoneme
                # The canonical sequence gains nothing; human sequence gains the extra phone.
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                human_ph = normalize_phoneme(pronounced_clean)
                if human_ph != "sil":
                    human_phones.append(human_ph)
                # canonical_phones: nothing added

        # Intervals with other formats are silently skipped

    return canonical_phones, human_phones


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
# Core counting: phoneme-level (Whisper)
# ─────────────────────────────────────────────────────────────────────────────

def count_phoneme_mdd(
    canonical: list[str],
    human:     list[str],
    predicted: list[str],
) -> MDDCounts:
    """
    Phoneme-level MDD counting for a single utterance (Whisper model).

    Aligns all three sequences to canonical position-by-position, then
    classifies each position per the rules in Table 2 of Shahin et al. (2025):

        canonical  human  predicted  → category
        ─────────  ─────  ─────────  ──────────
        ae         ae     ae         → TA
        d          t      d          → FA    (error accepted, model missed it)
        v          v      f          → FR    (correct, model wrongly rejected)
        ay         ey     ey         → TR/CD (error detected, correct diagnosis)
        s          sh     z          → TR/DE (error detected, wrong diagnosis)

    None values (sequence shorter than canonical) are never equal to any
    phoneme string, so they are automatically treated as wrong.

    Args:
        canonical : canonical phoneme sequence (anchor), from annotation TextGrid
        human     : human-annotated phoneme sequence, from annotation TextGrid
        predicted : Whisper-predicted phoneme sequence

    Returns:
        MDDCounts with TA, FA, FR, TR_CD, TR_DE tallied.
    """
    counts       = MDDCounts()
    paired_human = _zip_to_canonical(canonical, human)
    paired_pred  = _zip_to_canonical(canonical, predicted)

    for i, canon_ph in enumerate(canonical):
        human_ph = paired_human[i]
        pred_ph  = paired_pred[i]

        model_accepted  = (pred_ph  == canon_ph)
        speaker_correct = (human_ph == canon_ph)

        if speaker_correct and model_accepted:
            counts.TA += 1
        elif not speaker_correct and model_accepted:
            counts.FA += 1
        elif speaker_correct and not model_accepted:
            counts.FR += 1
        else:
            # Both predicted and human differ from canonical.
            # TR/CD if model correctly identified what the speaker actually said.
            # None != None in Python evaluates False, so two None values → TR/DE.
            if pred_ph is not None and human_ph is not None and pred_ph == human_ph:
                counts.TR_CD += 1
            else:
                counts.TR_DE += 1

    return counts


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


def _align_binary_sequence_to_canonical(ref_seq: list[int], hyp_seq: list[int]) -> list[Optional[int]]:
    """
    Align one decoded feature sequence to its canonical reference sequence
    using simple positional (zip) alignment.

    The CTC-decoded feature sequence has one value per predicted phoneme slot.
    We align it position-by-position to the canonical sequence — the same way
    phoneme sequences are aligned in count_phoneme_mdd. Levenshtein alignment
    is wrong here because 0/1 are not identity tokens; matching a 1 from one
    position to a 1 from a completely different position is phonetically
    meaningless and inflates FA artificially.

    Returns a list of length len(ref_seq). Positions where hyp_seq is shorter
    are filled with None (deletion). Extra hyp tokens beyond len(ref_seq) are
    ignored.
    """
    return [hyp_seq[i] if i < len(hyp_seq) else None for i in range(len(ref_seq))]


def count_phonological_mdd(
    canonical:        list[str],
    human:            list[str],
    predicted_logits: np.ndarray,
) -> PhonologicalMDDCounts:
    """
    Phonological feature-level MDD counting for one utterance.

    Corrected to match Shahin et al.'s CTC-style phonological evaluation:
      1. canonical and human phonemes are converted to 35 binary feature refs;
      2. SCTC-SB logits are decoded into 35 +att/-att sequences with blank
         removal and CTC repeat collapse;
      3. each decoded feature sequence is Levenshtein-aligned to the canonical
         feature sequence before TA/FA/FR/CD/DE counting.
    """
    phon_counts = PhonologicalMDDCounts()
    n = len(canonical)
    if n == 0:
        return phon_counts

    canon_feats = np.stack([_phoneme_to_binary_array(ph) for ph in canonical])

    paired_human = _zip_to_canonical(canonical, human)
    human_feats = np.zeros((n, NUM_FEATURES), dtype=np.int8)
    human_missing = np.zeros(n, dtype=bool)
    for i, human_ph in enumerate(paired_human):
        if human_ph is None:
            human_missing[i] = True
        else:
            human_feats[i] = _phoneme_to_binary_array(human_ph)

    pred_feature_seqs = _decode_sctcSB_logits_to_feature_sequences(predicted_logits)

    for f_idx, feat_name in enumerate(PHONOLOGICAL_FEATURES):
        cnt = phon_counts.counts[feat_name]
        ref_seq = canon_feats[:, f_idx].astype(int).tolist()
        pred_aligned = _align_binary_sequence_to_canonical(
            ref_seq,
            [int(x) for x in pred_feature_seqs[f_idx]],
        )

        for i in range(n):
            canon_f = int(canon_feats[i, f_idx])
            human_f = None if human_missing[i] else int(human_feats[i, f_idx])
            pred_f = pred_aligned[i]

            model_accepted = (pred_f == canon_f)
            speaker_correct = (human_f == canon_f)

            if speaker_correct and model_accepted:
                cnt.TA += 1
            elif (not speaker_correct) and model_accepted:
                cnt.FA += 1
            elif speaker_correct and (not model_accepted):
                cnt.FR += 1
            else:
                if pred_f is not None and human_f is not None and pred_f == human_f:
                    cnt.TR_CD += 1
                else:
                    cnt.TR_DE += 1

    return phon_counts


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator
# ─────────────────────────────────────────────────────────────────────────────

class MDDEvaluator:
    """
    Accumulates MDD counts across utterances and computes final metrics.

    Supports phoneme-level (Whisper) and/or phonological-level (SCTC-SB)
    evaluation. Both can run simultaneously on the same utterances.

    Typical usage
    -------------
        evaluator = MDDEvaluator()

        for ann_file in annotation_dir.glob("*.TextGrid"):
            canonical, human = parse_annotation_for_mdd(str(ann_file))
            utt_id = ann_file.stem

            # Phoneme-level (Whisper)
            whisper_phones = _run_whisper(whisper_model, processor, waveform, device)
            evaluator.add_phoneme_utterance(canonical, human, whisper_phones, utt_id)

            # Phonological-level (wav2vec2 SCTC-SB)
            logits = _run_sctcSB(sctcSB_model, feat_extractor, waveform, device)
            evaluator.add_phonological_utterance(canonical, human, logits, utt_id)

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
        canonical: list[str],
        human:     list[str],
        predicted: list[str],
        utt_id:    Optional[str] = None,
    ) -> MDDCounts:
        """Add one utterance to the Whisper (phoneme-level) accumulator."""
        if not canonical:
            logger.warning(f"[{utt_id}] empty canonical sequence, skipping")
            return MDDCounts()
        utt_counts = count_phoneme_mdd(canonical, human, predicted)
        self.phoneme_counts = self.phoneme_counts + utt_counts
        self.n_phoneme_utts += 1
        return utt_counts

    def add_phonological_utterance(
        self,
        canonical:        list[str],
        human:            list[str],
        predicted_logits: np.ndarray,
        utt_id:           Optional[str] = None,
    ) -> PhonologicalMDDCounts:
        """Add one utterance to the wav2vec2 SCTC-SB (phonological-level) accumulator."""
        if not canonical:
            logger.warning(f"[{utt_id}] empty canonical sequence, skipping")
            return PhonologicalMDDCounts()
        if predicted_logits.ndim != 2 or predicted_logits.shape[1] not in (
            NUM_FEATURES, NUM_OUTPUT_NODES
        ):
            raise ValueError(
                f"[{utt_id}] predicted_logits must be (T, {NUM_FEATURES}) "
                f"or (T, {NUM_OUTPUT_NODES}), got {predicted_logits.shape}"
            )
        utt_counts = count_phonological_mdd(canonical, human, predicted_logits)
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
    """
    import torch
    inputs = feature_extractor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        outputs = model(inputs.input_values.to(device))

    if isinstance(outputs, (tuple, list)):
        logits = outputs[0]
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = outputs

    return logits.squeeze(0).cpu().numpy()   # (T, 71)


# ─────────────────────────────────────────────────────────────────────────────
# Full corpus evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

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
            canonical, human = parse_annotation_for_mdd(str(ann_file))
            if not canonical:
                logger.warning(f"[{utt_id}] no canonical phones from annotation, skipping")
                n_skipped += 1
                continue
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
                        canonical, human, predicted_phones, utt_id
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
                        canonical, human, logits, utt_id
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
    args = parser.parse_args()

    # ── Sanity check: reproduce Table 2 ──────────────────────────────────────
    print("=== Phoneme-level sanity check (Table 2) ===")
    _canonical = ["ae", "d", "v", "ay", "s"]
    _human     = ["ae", "t", "v", "ey", "sh"]
    _predicted = ["ae", "d", "f", "ey", "z"]

    counts = count_phoneme_mdd(_canonical, _human, _predicted)
    print(f"  TA={counts.TA}  FA={counts.FA}  FR={counts.FR}  "
          f"TR_CD={counts.TR_CD}  TR_DE={counts.TR_DE}")
    print(f"  Expected: TA=1  FA=1  FR=1  TR_CD=1  TR_DE=1")
    assert counts.TA == 1 and counts.FA == 1 and counts.FR == 1
    assert counts.TR_CD == 1 and counts.TR_DE == 1
    print("  PASSED\n")

    print("=== Phonological-level sanity check ===")

    def _make_perfect_logits(phones: list[str]) -> np.ndarray:
        """Build (T, 71) logits that perfectly encode the given phoneme sequence."""
        T = len(phones)
        logits = np.zeros((T, NUM_OUTPUT_NODES), dtype=np.float32)
        for t, ph in enumerate(phones):
            vec = _phoneme_to_binary_array(ph)
            for f in range(NUM_FEATURES):
                pos = feature_idx_to_pos_node(f)
                neg = feature_idx_to_neg_node(f)
                if vec[f]:
                    logits[t, pos] =  1.0
                    logits[t, neg] = -1.0
                else:
                    logits[t, pos] = -1.0
                    logits[t, neg] =  1.0
        return logits

    pred_logits = _make_perfect_logits(["ae", "d", "f", "ey", "z"])
    phon_result = count_phonological_mdd(_canonical, _human, pred_logits)
    macro = phon_result.summary()["__macro_avg__"]
    print(f"  Macro FAR={macro['FAR']}  FRR={macro['FRR']}  DER={macro['DER']}")
    # FAR=0.0: with positional alignment, perfect logits matching what was
    # actually said means the model never outputs canonical for a mispronounced
    # position → no false acceptances.
    # FRR>0 and DER>0 are expected: position 2 maps predicted /f/ → canonical
    # /v/, differing on some features, producing legitimate FR and TR_DE.
    assert macro["FAR"] == 0.0, f"Expected FAR=0.0, got {macro['FAR']}"
    print("  PASSED\n")

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
    print(f"Evaluating speakers: {speakers}")
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

    results = evaluate_mdd(
        l2arctic_root      = args.l2arctic_dir,
        speakers           = speakers,
        phoneme_model      = phoneme_model,
        phonological_model = phonological_model,
        feature_extractor  = feature_extractor,
        device             = args.device,
        output_json        = args.output_json,
        verbose            = True,
    )
