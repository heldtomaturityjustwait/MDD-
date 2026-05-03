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
        blank_penalty: float = 2.0,
    ):
        super().__init__()
        self.num_features = num_features
        self.blank_idx = blank_idx
        self.reduction = reduction
        # blank_penalty: subtracted from blank logit before per-category softmax.
        # The shared blank receives gradient from all 35 categories simultaneously,
        # making it 35x stronger than any feature node. A penalty of ~2.0
        # counteracts this. Tune upward (3.0, 4.0) if blank_win_rate stays high.
        self.blank_penalty = blank_penalty
        self._build_category_node_maps()

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
        logits: torch.Tensor,           # (T, B, 71) — raw logits from model
        ctc_labels: List[List[List[int]]],  # [B][35][U_i] — label node indices
        input_lengths: torch.Tensor,    # (B,) — actual T for each item
    ) -> torch.Tensor:
        """
        Compute SCTC-SB loss.

        For each feature category i:
          1. Extract the 3-node slice [pos_i, neg_i, blank] from logits
          2. Apply softmax over those 3 nodes → log_softmax
          3. Re-map label indices to local {0=pos, 1=neg, 2=blank} space
          4. Compute standard CTC loss

        Total loss = (1/N) * sum_i CTC_loss_i    [matching paper Eq. 6 in log space]
        """
        T, B, _ = logits.shape
        total_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        for feat_idx in range(self.num_features):
            pos_node = feat_idx          # global index of +att_i
            neg_node = feat_idx + self.num_features   # global index of -att_i

            # ── Extract the 3-node logits for this category ───────────────
            # Shape: (T, B, 3)
            cat_logits = torch.stack([
                logits[:, :, pos_node],                              # +att
                logits[:, :, neg_node],                              # -att
                logits[:, :, self.blank_idx] - self.blank_penalty,  # shared blank (penalised)
            ], dim=-1)  # (T, B, 3)

            # Log-softmax over the 3 nodes
            cat_log_probs = F.log_softmax(cat_logits, dim=-1)  # (T, B, 3)

            # ── Build targets for this category ──────────────────────────
            # ctc_labels[b][feat_idx] is a list of GLOBAL node indices
            # We need LOCAL indices: pos→0, neg→1, blank→2
            local_targets = []
            target_lengths = []

            for b in range(B):
                global_seq = ctc_labels[b][feat_idx]  # list of ints
                local_seq = []
                for g in global_seq:
                    if g == pos_node:
                        local_seq.append(0)
                    elif g == neg_node:
                        local_seq.append(1)
                    else:
                        local_seq.append(2)   # blank (shouldn't appear in targets)
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
                log_probs=cat_log_probs,        # (T, B, 3)
                targets=padded_targets,          # (B, U)
                input_lengths=input_lengths,     # (B,)
                target_lengths=target_lengths_tensor,  # (B,)
                blank=2,                         # local blank index
                reduction=self.reduction,
                zero_infinity=True,              # avoid -inf on impossible alignments
            )

            total_loss = total_loss + ctc_loss

        # Average across features (log-space analog of Eq. 6)
        return total_loss / self.num_features
