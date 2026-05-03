"""
mdd_evaluation.py
=================
Mispronunciation Detection and Diagnosis (MDD) evaluation.

Supports two evaluation levels:
  1. Phoneme-level MDD      -- using fine-tuned Whisper
  2. Phonological-level MDD -- using wav2vec2 SCTC-SB (71-node output)

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
      From transcript/<utt_id>.txt → text_to_phones (CMUdict).
      The sentence the learner was supposed to pronounce.

  human-annotated
      What the speaker actually produced:
        1. annotation/<utt_id>.TextGrid  (if it exists) → parse_textgrid_annotation
           → actual_phones  (substitutions/deletions/insertions reflected)
        2. textgrid/<utt_id>.TextGrid  (fallback, no error labels)
           → parse_textgrid_canonical  (treated as all correct)

  predicted
      Whisper: output phoneme list  (list[str])
      wav2vec2 SCTC-SB: raw logits  (T, 71) numpy array

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
from dataset import (
    normalize_phoneme,
    text_to_phones,
    parse_textgrid_annotation,
    parse_textgrid_canonical,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence derivation
# ─────────────────────────────────────────────────────────────────────────────

def get_canonical_phones(transcript_path: str) -> list[str]:
    """
    Derive canonical phoneme sequence from a plain-text transcript file.

    Reads the sentence from transcript/<utt_id>.txt and converts it to
    phonemes via CMUdict (text_to_phones). This is the intended pronunciation
    and the anchor for all MDD evaluation.

    Args:
        transcript_path : path to <speaker>/transcript/<utt_id>.txt

    Returns:
        List of normalised CMU-39 phoneme strings, silences removed.
    """
    path = Path(transcript_path)
    if not path.exists():
        logger.warning(f"Transcript not found: {transcript_path}")
        return []
    text = path.read_text(encoding="utf-8").strip()
    phones = text_to_phones(text)
    return [p for p in phones if p != "sil"]


def get_human_phones(ann_tg_path: str, canon_tg_path: str) -> list[str]:
    """
    Derive the human-annotated phoneme sequence (what the speaker actually said).

    Priority:
      1. annotation/<utt_id>.TextGrid (if it exists):
         parse_textgrid_annotation → actual_phones
         Reflects real pronunciation: substituted phones used, deletions omitted,
         insertions included.
      2. textgrid/<utt_id>.TextGrid (fallback, no error labels):
         parse_textgrid_canonical → all phonemes treated as correct
         (human == canonical assumed for unannotated utterances).

    Args:
        ann_tg_path   : path to <speaker>/annotation/<utt_id>.TextGrid
        canon_tg_path : path to <speaker>/textgrid/<utt_id>.TextGrid  (fallback)

    Returns:
        List of normalised CMU-39 phoneme strings, silences removed.
    """
    if Path(ann_tg_path).exists():
        actual_phones, _, _ = parse_textgrid_annotation(ann_tg_path)
        actual_phones = [p for p in actual_phones if p != "sil"]
        if actual_phones:
            return actual_phones

    if Path(canon_tg_path).exists():
        logger.debug(
            f"No annotation at {ann_tg_path}; "
            f"using canonical TextGrid as human reference (all correct assumed)."
        )
        phones = parse_textgrid_canonical(canon_tg_path)
        return [p for p in phones if p != "sil"]

    return []


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
        canonical : canonical phoneme sequence from CMUdict (anchor)
        human     : human-annotated phoneme sequence
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
    Align one decoded feature sequence to its canonical reference sequence.

    Returns a list with length len(ref_seq). Insertions in hyp_seq are ignored
    because they do not correspond to a canonical phoneme position. Deletions are
    represented as None.
    """
    # local import avoids adding a new top-level dependency cycle
    from alignment import levenshtein_alignment

    _, _, _, ops = levenshtein_alignment(ref_seq, hyp_seq)
    aligned: list[Optional[int]] = []
    for op, ref_val, hyp_val in ops:
        if op == "I":
            continue
        if op == "D":
            aligned.append(None)
        else:  # C or S
            aligned.append(int(hyp_val))

    if len(aligned) < len(ref_seq):
        aligned.extend([None] * (len(ref_seq) - len(aligned)))
    return aligned[:len(ref_seq)]


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

        for utt_id, speaker in test_set:
            canonical = get_canonical_phones(f"{root}/{speaker}/transcript/{utt_id}.txt")
            human     = get_human_phones(
                ann_tg_path   = f"{root}/{speaker}/annotation/{utt_id}.TextGrid",
                canon_tg_path = f"{root}/{speaker}/textgrid/{utt_id}.TextGrid",
            )

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
            print(f"  PHONEME-LEVEL  (Whisper)  --  {self.n_phoneme_utts} utterances")
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

def _run_whisper(model, processor, waveform, device: str) -> list[str]:
    """
    Run fine-tuned Whisper → predicted phoneme list.

    Expects Whisper to output space-separated ARPAbet tokens (e.g. "AH0 D V AY1 S").
    Returns normalised CMU-39 phoneme strings, silences removed.
    """
    import torch
    inputs = processor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt"
    )
    with torch.no_grad():
        generated = model.generate(inputs.input_features.to(device))
    tokens = processor.batch_decode(generated, skip_special_tokens=True)
    if not tokens:
        return []
    phones = [normalize_phoneme(p) for p in tokens[0].strip().split()]
    return [p for p in phones if p != "sil"]


def _run_sctcSB(model, feature_extractor, waveform, device: str) -> np.ndarray:
    """
    Run SCTC-SB wav2vec2 → raw logits (T, 71) as a numpy array.

    Supports PhonologicalWav2Vec2 (returns tuple) and HuggingFace-style
    models (expose a .logits attribute).
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
    l2arctic_root:           str,
    speakers:                list[str],
    whisper_model            = None,
    whisper_processor        = None,
    sctcSB_model             = None,
    sctcSB_feature_extractor = None,
    device:      str           = "cuda",
    output_json: Optional[str] = None,
    verbose:     bool          = True,
) -> dict:
    """
    Run full MDD evaluation over L2-ARCTIC scripted utterances.

    Directory structure assumed per speaker:
        <speaker>/
          wav/           ← .wav files
          transcript/    ← <utt_id>.txt  (plain-text sentence for CMUdict)
          annotation/    ← <utt_id>.TextGrid  (human error annotation, ~15% coverage)
          textgrid/      ← <utt_id>.TextGrid  (force-aligned, all utterances, fallback)

    For each utterance:
      canonical  ← transcript/<utt>.txt  →  text_to_phones (CMUdict)
      human      ← annotation/<utt>.TextGrid  OR  textgrid/<utt>.TextGrid fallback
      predicted  ← Whisper output  and/or  SCTC-SB logits

    Utterances with no canonical phones or no human phones are skipped.

    Args:
        l2arctic_root            : root path of the L2-ARCTIC corpus
        speakers                 : list of speaker IDs to evaluate
        whisper_model            : fine-tuned WhisperForConditionalGeneration (optional)
        whisper_processor        : WhisperProcessor (optional)
        sctcSB_model             : SCTC-SB PhonologicalWav2Vec2 model (optional)
        sctcSB_feature_extractor : Wav2Vec2FeatureExtractor (optional)
        device                   : "cuda" or "cpu"
        output_json              : if set, save results to this path as JSON
        verbose                  : log progress every 50 utterances

    Returns:
        dict with keys: phoneme_level, phonological_level, counts
    """
    import torch
    import torchaudio

    evaluator = MDDEvaluator()
    root      = Path(l2arctic_root)

    if whisper_model is not None:
        whisper_model.to(device).eval()
    if sctcSB_model is not None:
        sctcSB_model.to(device).eval()

    n_total = n_skipped = 0

    for spk in speakers:
        spk_dir = root / spk
        wav_dir = spk_dir / "wav"
        trs_dir = spk_dir / "transcript"
        ann_dir = spk_dir / "annotation"
        tg_dir  = spk_dir / "textgrid"

        if not wav_dir.exists():
            logger.warning(f"wav dir not found for speaker {spk}, skipping")
            continue

        wav_files = sorted(wav_dir.glob("*.wav"))
        for wav_idx, wav_file in enumerate(wav_files):
            utt_id = wav_file.stem
            n_total += 1

            if verbose and wav_idx % 50 == 0:
                print(f"  [{spk}] {wav_idx + 1}/{len(wav_files)}  {utt_id}")

            # ── 1. Canonical: transcript → CMUdict ────────────────────────
            canonical = get_canonical_phones(str(trs_dir / f"{utt_id}.txt"))
            if not canonical:
                logger.warning(f"[{utt_id}] no canonical phones, skipping")
                n_skipped += 1
                continue

            # ── 2. Human: annotation TextGrid → textgrid fallback ─────────
            human = get_human_phones(
                ann_tg_path   = str(ann_dir / f"{utt_id}.TextGrid"),
                canon_tg_path = str(tg_dir  / f"{utt_id}.TextGrid"),
            )
            if not human:
                logger.warning(f"[{utt_id}] no human phones, skipping")
                n_skipped += 1
                continue

            # ── 3. Load audio ──────────────────────────────────────────────
            waveform, sr = torchaudio.load(str(wav_file))
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.mean(dim=0)   # mono (T,)

            # ── 4. Phoneme-level MDD (Whisper) ─────────────────────────────
            if whisper_model is not None and whisper_processor is not None:
                try:
                    predicted_phones = _run_whisper(
                        whisper_model, whisper_processor, waveform, device
                    )
                    evaluator.add_phoneme_utterance(
                        canonical, human, predicted_phones, utt_id
                    )
                except Exception as e:
                    logger.warning(f"[{utt_id}] Whisper inference error: {e}")

            # ── 5. Phonological-level MDD (SCTC-SB wav2vec2) ───────────────
            if sctcSB_model is not None and sctcSB_feature_extractor is not None:
                try:
                    logits = _run_sctcSB(
                        sctcSB_model, sctcSB_feature_extractor, waveform, device
                    )
                    evaluator.add_phonological_utterance(
                        canonical, human, logits, utt_id
                    )
                except Exception as e:
                    logger.warning(f"[{utt_id}] SCTC-SB inference error: {e}")

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
        description="MDD evaluation for phoneme-level (Whisper) and/or "
                    "phonological-level (SCTC-SB wav2vec2) models."
    )
    parser.add_argument("--l2arctic_dir",  type=str, default=None)
    parser.add_argument("--sctcSB_model",  type=str, default=None,
                        help="Path to SCTC-SB checkpoint (.pt)")
    parser.add_argument("--whisper_model", type=str, default=None,
                        help="Path to fine-tuned Whisper checkpoint directory")
    parser.add_argument("--speakers",      type=str, nargs="+", default=None)
    parser.add_argument("--output_json",   type=str, default=None)
    parser.add_argument("--device",        type=str, default="cuda")
    parser.add_argument("--sanity_check",  action="store_true")
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

    # Simulate wav2vec2 predicting ["ae","d","f","ey","z"] — same as Whisper in Table 2
    pred_logits = _make_perfect_logits(["ae", "d", "f", "ey", "z"])
    phon_result = count_phonological_mdd(_canonical, _human, pred_logits)
    macro = phon_result.summary()["__macro_avg__"]
    print(f"  Macro FAR={macro['FAR']}  FRR={macro['FRR']}  DER={macro['DER']}")
    # A perfect predictor has no diagnosis errors (DER == 0)
    assert macro["DER"] == 0.0, f"Expected DER=0.0, got {macro['DER']}"
    print("  PASSED\n")

    if args.sanity_check:
        sys.exit(0)

    # ── Full corpus evaluation ────────────────────────────────────────────────
    if args.l2arctic_dir is None:
        parser.error("--l2arctic_dir is required for corpus evaluation.")
    if args.sctcSB_model is None and args.whisper_model is None:
        parser.error("Provide at least one of --sctcSB_model or --whisper_model.")

    DEFAULT_TEST_SPEAKERS = ["RRBI", "YBAA", "HJK", "BWC", "EBVS", "YDCK"]
    speakers = args.speakers if args.speakers else DEFAULT_TEST_SPEAKERS
    print(f"Evaluating speakers: {speakers}")
    print(f"Device: {args.device}\n")

    sctcSB_model             = None
    sctcSB_feature_extractor = None
    whisper_model            = None
    whisper_processor        = None

    if args.sctcSB_model:
        import torch
        from transformers import Wav2Vec2FeatureExtractor
        from wav2vec2_phonological import PhonologicalWav2Vec2
        print(f"[SCTC-SB] Loading checkpoint: {args.sctcSB_model}")
        sctcSB_model = PhonologicalWav2Vec2()
        state = torch.load(args.sctcSB_model, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        sctcSB_model.load_state_dict(state)
        sctcSB_model.eval()
        sctcSB_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-large-robust"
        )
        print("[SCTC-SB] Ready.")

    if args.whisper_model:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        print(f"[Whisper] Loading checkpoint: {args.whisper_model}")
        whisper_model     = WhisperForConditionalGeneration.from_pretrained(
            args.whisper_model
        )
        whisper_processor = WhisperProcessor.from_pretrained(args.whisper_model)
        whisper_model.eval()
        print("[Whisper] Ready.")

    results = evaluate_mdd(
        l2arctic_root            = args.l2arctic_dir,
        speakers                 = speakers,
        whisper_model            = whisper_model,
        whisper_processor        = whisper_processor,
        sctcSB_model             = sctcSB_model,
        sctcSB_feature_extractor = sctcSB_feature_extractor,
        device                   = args.device,
        output_json              = args.output_json,
        verbose                  = True,
    )
