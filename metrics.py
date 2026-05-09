"""
metrics.py
==========
Evaluation metrics strictly following paper Section 5.3.

5.3.1 Phonological feature recognition:
    FER = (S + D + I) / N          (Eq. 8)
    Precision, Recall, F1           (Eq. 9)
"""

from dataclasses import dataclass
import numpy as np

from phonological_features import PHONOLOGICAL_FEATURES, NUM_FEATURES
from alignment import levenshtein_alignment


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
