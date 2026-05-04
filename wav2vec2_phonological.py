"""
wav2vec2_phonological.py
========================
Phonological feature detection model strictly following Figure 2 of
Shahin et al. (Speech Communication, 2025).

Architecture (Fig. 2):
    Raw Speech
        │
        ▼
    wav2vec2.0 (pre-trained, CNN encoder frozen)
        ├─ CNN Feature Extractor   [FROZEN]
        └─ Transformer             [FINE-TUNED]
        │
        ▼
    Linear Layer  (hidden_size → 71 nodes)
        │
        ▼
    SCTC-SB Loss (during training)
    OR
    argmax per category (during inference)

Output nodes:
    0..34   → +att_i  (presence of feature i)
    35..69  → -att_i  (absence of feature i)
    70      → shared blank
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from typing import Optional

from phonological_features import NUM_FEATURES, NUM_OUTPUT_NODES, BLANK_IDX


class PhonologicalWav2Vec2(nn.Module):
    """
    wav2vec2-based phonological feature detection model.

    Args:
        pretrained_model_name (str): HuggingFace model ID.
            Paper uses 'facebook/wav2vec2-large-robust' (best performing).
        num_output_nodes (int): 71 = 35(+att) + 35(-att) + 1(blank).
        freeze_cnn_encoder (bool): Whether to freeze CNN feature extractor.
            Paper Section 5.2: "its parameters were fixed during fine-tuning".
    """

    def __init__(
        self,
        pretrained_model_name: str = "facebook/wav2vec2-large-robust",
        num_output_nodes: int = NUM_OUTPUT_NODES,
        freeze_cnn_encoder: bool = True,
    ):
        super().__init__()
        self.num_output_nodes = num_output_nodes
        self.num_features = NUM_FEATURES
        self.blank_idx = BLANK_IDX

        # ── Load pre-trained wav2vec2 ─────────────────────────────────────
        print(f"[PhonologicalWav2Vec2] Loading '{pretrained_model_name}' ...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        # ── Freeze CNN encoder (feature extractor) ────────────────────────
        # Paper: "Except for the CNN encoder layer, the whole network was
        # then fine-tuned"
        if freeze_cnn_encoder:
            self.wav2vec2.feature_extractor._freeze_parameters()
            print("[PhonologicalWav2Vec2] CNN encoder FROZEN.")

        # ── Linear projection head (Fig. 2) ──────────────────────────────
        # "A linear layer was added on top of the transformer module with
        #  number of nodes equals to the number of target phonological-features"
        hidden_size = self.wav2vec2.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_output_nodes)

        # ── Blank node bias initialization ────────────────────────────────
        # The shared blank node (index 70) appears in all 35 category losses,
        # so it receives 35x more gradient than any feature node. Without
        # correction, CTC collapses to predicting blank at every timestep
        # within the first epoch. Initializing the blank bias to -4.0 forces
        # the model to earn blank probability rather than defaulting to it.
        with torch.no_grad():
            self.classifier.bias[self.blank_idx].fill_(-2.5)

        print(f"[PhonologicalWav2Vec2] hidden_size={hidden_size}, "
              f"output_nodes={num_output_nodes}")

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            logits        : (B, T_frames, 71)  raw logits
            output_lengths: (B,)  number of valid frames per batch item
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)

        if attention_mask is not None:
            output_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            )
        else:
            B, T_audio = input_values.shape
            ones = torch.ones(B, dtype=torch.long, device=input_values.device) * T_audio
            output_lengths = self._get_feat_extract_output_lengths(ones)

        return logits, output_lengths

    def _get_feat_extract_output_lengths(
        self, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        return self.wav2vec2._get_feat_extract_output_lengths(input_lengths)

    @torch.no_grad()
    def decode(
        self,
        logits: torch.Tensor,    # (B, T, 71) or (T, 71) for single item
    ) -> list[list[list[int]]]:
        """
        Greedy CTC decoding per category.

        For each feature category i, applies argmax over the 3-node slice
        [pos_i, neg_i, blank] and collapses repeated labels + blanks.

        Paper Section 3.3, Eq. 7:
            h_i(x) = argmax_j y^t_{i,j}

        Returns:
            decoded: [B][35]  list of decoded label sequences
                     Each label sequence contains +att(True) or -att(False)
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        B, T, _ = logits.shape
        decoded_batch = []

        for b in range(B):
            decoded_features = []
            for feat_idx in range(self.num_features):
                pos_node = feat_idx
                neg_node = feat_idx + self.num_features

                # Extract 3-node slice: (T, 3)
                cat_logits = torch.stack([
                    logits[b, :, pos_node],
                    logits[b, :, neg_node],
                    logits[b, :, self.blank_idx],
                ], dim=-1)  # (T, 3)

                # Argmax: 0=+att, 1=-att, 2=blank
                preds = cat_logits.argmax(dim=-1)  # (T,)

                # CTC collapse: remove blanks and repeated labels
                collapsed = []
                prev = -1
                for p in preds.tolist():
                    if p == 2:       # blank
                        prev = -1
                        continue
                    if p != prev:
                        collapsed.append(p == 0)   # True=+att, False=-att
                        prev = p

                decoded_features.append(collapsed)
            decoded_batch.append(decoded_features)

        return decoded_batch

    def count_parameters(self) -> dict:
        """Count trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}


# ─────────────────────────────────────────────────────────────────────────────
# Phoneme-level baseline model (for comparison, paper Section 3)
# Same architecture but with 40 output nodes (39 phonemes + 1 blank)
# and standard CTC loss
# ─────────────────────────────────────────────────────────────────────────────
class PhonemeLevelWav2Vec2(nn.Module):
    """
    Phoneme-level MDD baseline (paper Section 3, Fig. 1 top branch).
    Uses standard CTC with 39 phonemes + blank.
    """

    def __init__(
        self,
        pretrained_model_name: str = "facebook/wav2vec2-large-robust",
        num_phonemes: int = 39,
        freeze_cnn_encoder: bool = True,
    ):
        super().__init__()
        self.num_phonemes = num_phonemes
        self.blank_idx = num_phonemes   # index 39 = blank

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        if freeze_cnn_encoder:
            self.wav2vec2.feature_extractor._freeze_parameters()

        hidden_size = self.wav2vec2.config.hidden_size
        # 40 nodes: 39 phonemes + 1 blank
        self.classifier = nn.Linear(hidden_size, num_phonemes + 1)

        # ── Blank node bias initialization ────────────────────────────────
        # Same CTC blank collapse problem as the phonological model.
        # Initialize blank bias (index 39) to -4.0 to prevent the model
        # from defaulting to all-blank predictions early in training.
        with torch.no_grad():
            self.classifier.bias[self.blank_idx].fill_(-1)

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.classifier(hidden_states)

        if attention_mask is not None:
            output_lengths = self.wav2vec2._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1))
        else:
            B, T = input_values.shape
            ones = torch.ones(B, dtype=torch.long, device=input_values.device) * T
            output_lengths = self.wav2vec2._get_feat_extract_output_lengths(ones)

        return logits, output_lengths
