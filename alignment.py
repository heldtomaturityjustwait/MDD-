"""
alignment.py
============
Levenshtein-based sequence alignment for MDD evaluation.

Paper Section 5.3:
  "The Levenshtein distance metric works by measuring the difference
   between two sequences in terms of the number of Insertion (I),
   Deletion (D), and Substitution (S) edits."
"""

from typing import Any


def levenshtein_alignment(
    ref: list[Any],
    hyp: list[Any],
) -> tuple[int, int, int, list[tuple[str, Any, Any]]]:
    """
    Compute Levenshtein alignment between reference and hypothesis sequences.

    Returns:
        S  : number of substitutions
        D  : number of deletions
        I  : number of insertions
        ops: list of (op_type, ref_item, hyp_item)
             op_type ∈ {'C'=correct, 'S'=substitution, 'D'=deletion, 'I'=insertion}
    """
    n = len(ref)
    m = len(hyp)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # deletion
                    dp[i][j - 1],     # insertion
                    dp[i - 1][j - 1], # substitution
                )

    # Traceback
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            ops.append(("C", ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("S", ref[i - 1], hyp[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1

    ops.reverse()

    S = sum(1 for op in ops if op[0] == "S")
    D = sum(1 for op in ops if op[0] == "D")
    I = sum(1 for op in ops if op[0] == "I")

    return S, D, I, ops


def compute_fer(ref: list[Any], hyp: list[Any]) -> float:
    """
    Feature Error Rate = (S + D + I) / N  (paper Eq. 8)
    """
    N = len(ref)
    if N == 0:
        return 0.0
    S, D, I, _ = levenshtein_alignment(ref, hyp)
    return (S + D + I) / N
