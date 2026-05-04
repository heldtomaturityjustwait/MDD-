"""
sctc_loss.py
============
Separable CTC with Shared Blank (SCTC-SB) loss function.

Paper Section 3.2:
  "In this work, we adopted the SCTC approach as the objective is the
   classification of the separate phonological feature categories.
   However, to maintain the alignment between components, one blank node
   was shared among all categories."

  Final objective (Eq. 6):
      p(l|x) = ∏_{i=1}^{N} p(l_i | x)

  i.e. the loss is the SUM of per-category CTC losses (negative log of
  product = sum of negative logs).

Output node layout (71 nodes):
  nodes 0..34   → +att_i   (presence of feature i)
  nodes 35..69  → -att_i   (absence of feature i)
  node  70      → SHARED blank

For each category i, the valid label alphabet is {+att_i, -att_i, blank}.
The softmax is applied over ONLY these 3 nodes when computing CTC
probabilities for category i.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

NUM_FEATURES = 35
BLANK_IDX = 70   # shared blank node index


class SCTCSBLoss(nn.Module):
    """
    Separable CTC with Shared Blank (SCTC-SB).

    Args:
        num_features (int): N=35, number of phonological feature categories.
        blank_idx (int): Index of the shared blank output node (70).
        reduction (str): 'mean' or 'sum' across the batch.
    """
    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        blank_idx: int = BLANK_IDX,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_features = num_features
        self.blank_idx = blank_idx
        self.reduction = reduction

    def _build_category_node_maps(self):
        """Pre-compute the 3 relevant node indices for each category."""
        # category_nodes[i] = [pos_node_i, neg_node_i, blank_node]
        self.register_buffer(
            "category_pos_nodes",
            torch.arange(self.num_features, dtype=torch.long),  # 0..34
        )
        self.register_buffer(
            "category_neg_nodes",
            torch.arange(self.num_features, dtype=torch.long) + self.num_features,  # 35..69
        )

    def forward(
        self,
        logits: torch.Tensor,
        ctc_labels: List[List[List[int]]],
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        T, B, _ = logits.shape
        total_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        for feat_idx in range(self.num_features):
            pos_node = feat_idx
            neg_node = feat_idx + self.num_features

            # Fix 1: no blank scaling
            cat_logits = torch.stack([
                logits[:, :, pos_node],
                logits[:, :, neg_node],
                logits[:, :, self.blank_idx],
            ], dim=-1)  # (T, B, 3)

            cat_log_probs = F.log_softmax(cat_logits, dim=-1)  # (T, B, 3)

            local_targets = []
            target_lengths = []

            for b in range(B):
                global_seq = ctc_labels[b][feat_idx]
                local_seq = []
                for g in global_seq:
                    if g == pos_node:
                        local_seq.append(0)
                    elif g == neg_node:
                        local_seq.append(1)
                    else:
                        # Fix 3: crash loudly instead of silently corrupting
                        raise ValueError(
                            f"Unexpected global label {g} for feature {feat_idx} "
                            f"(expected {pos_node} or {neg_node})"
                        )
                local_targets.append(torch.tensor(local_seq, dtype=torch.long,
                                                device=logits.device))
                target_lengths.append(len(local_seq))

            max_target_len = max(target_lengths) if target_lengths else 1
            padded_targets = torch.zeros(B, max_target_len, dtype=torch.long,
                                        device=logits.device)
            for b, t in enumerate(local_targets):
                if len(t) > 0:
                    padded_targets[b, :len(t)] = t

            target_lengths_tensor = torch.tensor(
                target_lengths, dtype=torch.long, device=logits.device
            )

            ctc_loss = F.ctc_loss(
                log_probs=cat_log_probs,
                targets=padded_targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths_tensor,
                blank=2,
                reduction=self.reduction,  # 'mean' — stable across batch sizes
                zero_infinity=True,
            )

            total_loss = total_loss + ctc_loss

        # Fix 2: sum across features, matching Eq. 6 exactly
        return total_loss
    