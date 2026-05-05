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

  In log space (what we actually compute):
      loss = -log p(l|x) = sum_{i=1}^{N} -log p(l_i|x)
           = sum_{i=1}^{N} CTC_loss_i

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
        # Fix 4: removed dead _build_category_node_maps() buffers

    def forward(
        self,
        logits: torch.Tensor,               # (T, B, 71) — raw logits from model
        ctc_labels: List[List[List[int]]],  # [B][35][U_i] — label node indices
        input_lengths: torch.Tensor,        # (B,) — actual frame lengths
    ) -> torch.Tensor:
        """
        Compute SCTC-SB loss.

        For each feature category i:
          1. Extract the 3-node slice [pos_i, neg_i, blank] from logits
          2. Apply log_softmax over those 3 nodes
          3. Re-map label indices to local {0=+att, 1=-att, 2=blank} space
          4. Compute standard CTC loss

        Total loss = sum_i CTC_loss_i    [Eq. 6 in log space]
        """
        T, B, _ = logits.shape
        total_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        for feat_idx in range(self.num_features):
            pos_node = feat_idx                       # global index of +att_i
            neg_node = feat_idx + self.num_features   # global index of -att_i

            # Extract the 3-node logits for this category.
            # Fix 1: no blank scaling — use raw blank logit as-is.
            # Paper Section 3.2: softmax over C'_i = {+att_i, -att_i, blank}
            # with no modification to any logit.
            cat_logits = torch.stack([
                logits[:, :, pos_node],        # local index 0 = +att_i
                logits[:, :, neg_node],        # local index 1 = -att_i
                logits[:, :, self.blank_idx],  # local index 2 = blank
            ], dim=-1)  # (T, B, 3)

            # Log-softmax over the 3 nodes (paper Eq. 4-5)
            cat_log_probs = F.log_softmax(cat_logits, dim=-1)  # (T, B, 3)

            # Build targets for this category
            local_targets = []
            target_lengths = []

            for b in range(B):
                global_seq = ctc_labels[b][feat_idx]  # list of global node indices
                local_seq = []
                for g in global_seq:
                    if g == pos_node:
                        local_seq.append(0)   # +att → local 0
                    elif g == neg_node:
                        local_seq.append(1)   # -att → local 1
                    else:
                        # Fix 3: crash loudly — blanks must never appear in
                        # CTC targets. If this fires, the label preparation
                        # upstream has a bug.
                        raise ValueError(
                            f"Unexpected global label {g} for feature {feat_idx} "
                            f"(expected pos={pos_node} or neg={neg_node}). "
                            f"Blank tokens must not appear in CTC target sequences."
                        )
                local_targets.append(torch.tensor(local_seq, dtype=torch.long,
                                                   device=logits.device))
                target_lengths.append(len(local_seq))

            # Pad targets for batched CTC
            max_target_len = max(target_lengths) if target_lengths else 1
            padded_targets = torch.zeros(
                B, max_target_len, dtype=torch.long, device=logits.device
            )
            for b, t in enumerate(local_targets):
                if len(t) > 0:
                    padded_targets[b, :len(t)] = t

            target_lengths_tensor = torch.tensor(
                target_lengths, dtype=torch.long, device=logits.device
            )

            # Blank in local space = index 2
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

        # Fix 2: return raw sum, matching Eq. 6 exactly.
        # Do NOT divide by num_features — Eq. 6 is a product (sum in log space),
        # not an average.
        return total_loss
