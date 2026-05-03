"""
metrics.py
==========
Evaluation metrics strictly following paper Section 5.3.

5.3.1 Phonological feature recognition:
    FER = (S + D + I) / N          (Eq. 8)
    Precision, Recall, F1           (Eq. 9)

5.3.2 MDD performance:
    FAR = FA / (FA + TR)           (Eq. 10)
    FRR = FR / (FR + TA)           (Eq. 10)
    DER = DE / (CD + DE)           (Eq. 10)

Where:
    TA = True Acceptance  (correct phoneme → recognized as canonical)
    TR = True Rejection   (mispronounced  → recognized differently)
    FA = False Acceptance (mispronounced  → recognized as canonical)
    FR = False Rejection  (correct phoneme→ recognized differently)
    CD = Correctly Diagnosed (TR where recognized == actually pronounced)
    DE = Diagnosis Error     (TR where recognized != actually pronounced)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from phonological_features import PHONOLOGICAL_FEATURES, NUM_FEATURES
from alignment import levenshtein_alignment, compute_fer


# ─────────────────────────────────────────────────────────────────────────────
# 5.3.1  Phonological Feature Recognition Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureMetrics:
    feature_name: str
    TP: int = 0   # +att predicted, +att reference
    FP: int = 0   # +att predicted, -att reference
    TN: int = 0   # -att predicted, -att reference
    FN: int = 0   # -att predicted, +att reference
    S:  int = 0   # substitutions in sequence
    D:  int = 0   # deletions
    I:  int = 0   # insertions
    N:  int = 0   # reference length

    @property
    def precision(self) -> float:
        return self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fer(self) -> float:
        return (self.S + self.D + self.I) / self.N if self.N > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return 1.0 - self.fer


def update_feature_metrics(
    metrics: FeatureMetrics,
    ref_seq: list[bool],   # True=+att, False=-att
    hyp_seq: list[bool],
) -> None:
    """Update FeatureMetrics in-place given ref and hyp binary sequences."""
    S, D, I, ops = levenshtein_alignment(ref_seq, hyp_seq)
    metrics.S += S
    metrics.D += D
    metrics.I += I
    metrics.N += len(ref_seq)

    for op, ref_val, hyp_val in ops:
        if op == "C":
            if ref_val:  metrics.TP += 1
            else:        metrics.TN += 1
        elif op == "S":
            if hyp_val:  metrics.FP += 1   # predicted +att but ref is -att
            else:        metrics.FN += 1   # predicted -att but ref is +att


def compute_all_feature_metrics(
    all_ref_seqs: list[list[list[bool]]],   # [N_utts][35][U]
    all_hyp_seqs: list[list[list[bool]]],   # [N_utts][35][U]
) -> list[FeatureMetrics]:
    """
    Compute per-feature metrics over a dataset.

    Returns list of 35 FeatureMetrics, one per feature.
    """
    metrics_list = [
        FeatureMetrics(feature_name=PHONOLOGICAL_FEATURES[i])
        for i in range(NUM_FEATURES)
    ]

    for ref_utt, hyp_utt in zip(all_ref_seqs, all_hyp_seqs):
        for feat_idx in range(NUM_FEATURES):
            update_feature_metrics(
                metrics_list[feat_idx],
                ref_utt[feat_idx],
                hyp_utt[feat_idx],
            )

    return metrics_list


def print_feature_metrics(metrics_list: list[FeatureMetrics]) -> None:
    print(f"\n{'Feature':<16} {'ACC%':>7} {'FER%':>7} {'PRE%':>7} {'REC%':>7} {'F1%':>7}")
    print("-" * 55)
    accs, fers, pres, recs, f1s = [], [], [], [], []
    for m in metrics_list:
        acc = m.accuracy * 100
        fer = m.fer * 100
        pre = m.precision * 100
        rec = m.recall * 100
        f1  = m.f1 * 100
        print(f"{m.feature_name:<16} {acc:>7.2f} {fer:>7.2f} {pre:>7.2f} {rec:>7.2f} {f1:>7.2f}")
        accs.append(acc); fers.append(fer); pres.append(pre)
        recs.append(rec); f1s.append(f1)
    print("-" * 55)
    print(f"{'AVERAGE':<16} {np.mean(accs):>7.2f} {np.mean(fers):>7.2f} "
          f"{np.mean(pres):>7.2f} {np.mean(recs):>7.2f} {np.mean(f1s):>7.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.3.2  MDD Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MDDCounts:
    """Running counters for MDD evaluation (paper Section 5.3.2)."""
    TA: int = 0   # True Acceptance
    TR: int = 0   # True Rejection
    FA: int = 0   # False Acceptance
    FR: int = 0   # False Rejection
    CD: int = 0   # Correctly Diagnosed (subset of TR)
    DE: int = 0   # Diagnosis Error     (subset of TR)

    @property
    def FAR(self) -> float:
        denom = self.FA + self.TR
        return self.FA / denom if denom > 0 else 0.0

    @property
    def FRR(self) -> float:
        denom = self.FR + self.TA
        return self.FR / denom if denom > 0 else 0.0

    @property
    def DER(self) -> float:
        denom = self.CD + self.DE
        return self.DE / denom if denom > 0 else 0.0

    def __add__(self, other: "MDDCounts") -> "MDDCounts":
        return MDDCounts(
            TA=self.TA + other.TA, TR=self.TR + other.TR,
            FA=self.FA + other.FA, FR=self.FR + other.FR,
            CD=self.CD + other.CD, DE=self.DE + other.DE,
        )

    def summary(self) -> str:
        return (f"FAR={self.FAR*100:.2f}% | FRR={self.FRR*100:.2f}% | "
                f"DER={self.DER*100:.2f}%  "
                f"(TA={self.TA} TR={self.TR} FA={self.FA} FR={self.FR} "
                f"CD={self.CD} DE={self.DE})")


def evaluate_phonological_mdd(
    mdd_records: list[dict],
    decoded_features: list[list[bool]],    # [35][U_hyp]
    ref_feature_seqs: list[list[bool]],    # [35][U_ref]
) -> MDDCounts:
    """
    Compute MDD counts for one utterance at the phonological level.

    Paper Section 5.3.2:
      "Similar metrics were used for the phonological-level MDD.
       For instance, if the phoneme /s/ was mispronounced as /z/,
       this was considered a mispronunciation of the voiced feature only
       and correct pronunciation for all other phonological features."

    Args:
        mdd_records: parsed annotation (list of {canonical, status, pronounced})
        decoded_features: model output [35][U_hyp] bool lists
        ref_feature_seqs: reference [35][U_ref] bool lists
    """
    counts = MDDCounts()

    for feat_idx in range(NUM_FEATURES):
        ref_seq = ref_feature_seqs[feat_idx]
        hyp_seq = decoded_features[feat_idx]

        # Align
        _, _, _, ops = levenshtein_alignment(ref_seq, hyp_seq)

        # We need MDD-level ops: match to annotation records
        # For simplicity, use the alignment ops directly
        ref_pos = 0
        for op, ref_val, hyp_val in ops:
            if ref_pos >= len(mdd_records):
                break

            record = mdd_records[ref_pos]
            is_mispronounced = record["status"] != "C"

            if op == "C":
                if is_mispronounced:
                    counts.FA += 1   # error accepted as correct
                else:
                    counts.TA += 1   # correct accepted
                ref_pos += 1
            elif op == "S":
                if is_mispronounced:
                    # True Rejection: did the model diagnose correctly?
                    counts.TR += 1
                    # For phonological MDD: CD if the feature changed correctly
                    if ref_val != hyp_val:   # feature changed → detected
                        counts.CD += 1
                    else:
                        counts.DE += 1
                else:
                    counts.FR += 1   # correct pronunciation rejected
                ref_pos += 1
            elif op == "D":
                if is_mispronounced:
                    counts.TR += 1; counts.DE += 1
                else:
                    counts.FR += 1
                ref_pos += 1
            # Insertion: extra hypothesis symbol, no canonical reference consumed

    return counts


def print_mdd_summary(counts: MDDCounts, label: str = "MDD") -> None:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  FAR : {counts.FAR*100:.2f}%")
    print(f"  FRR : {counts.FRR*100:.2f}%")
    print(f"  DER : {counts.DER*100:.2f}%")
    print(f"  TA={counts.TA}  TR={counts.TR}  FA={counts.FA}  FR={counts.FR}")
    print(f"  CD={counts.CD}  DE={counts.DE}")
    print(f"{'='*50}\n")
