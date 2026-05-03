"""
train_phoneme.py
================
Phoneme-level CTC baseline using wav2vec2-large-robust.

Comparison model for the phonological-level SCTC-SB system.
Same backbone, same data, same train/test split.
Only the output representation differs:

  Phoneme-level CTC   : Linear(1024 → 40) + standard CTCLoss
  Phonological SCTC-SB: Linear(1024 → 71) + SCTC-SB loss  ← main model

=============================================================================
EVALUATION DESIGN
=============================================================================

Task 1 — Feature Recognition (FER / ACC / F1 per feature):
  The phoneme model outputs recognized phoneme sequences.
  Each recognized phoneme is mapped to its 35 binary feature values
  via the lookup table in phonological_features.py.
  This gives a [35][U] feature sequence — exactly the same format as
  the SCTC-SB model output. The same compute_all_feature_metrics()
  function from metrics.py is then applied, producing identical output
  format to train.py for direct comparison.

  Reference  : actual_phones → feature sequences  (same as train.py)
  Hypothesis : recognized_phones → feature sequences (via lookup)
  Metrics    : FER, ACC, Precision, Recall, F1 per feature
               printed by print_feature_metrics() — identical to train.py

Task 2 — Phoneme Error Rate (PER):
  Standard phoneme recognition accuracy reported alongside feature
  metrics for completeness.
  Reference: actual_phones. Hypothesis: CTC greedy decode.
  Metric: PER = (S + D + I) / N via Levenshtein.

=============================================================================

Usage:
    python train_phoneme.py \
        --config    config.yaml \
        --timit_dir /path/to/timit/data \
        --data_dir  /path/to/l2arctic \
        --output_dir /path/to/phoneme_output

    # Resume:
    python train_phoneme.py \
        --config    config.yaml \
        --data_dir  /path/to/l2arctic \
        --output_dir /path/to/phoneme_output \
        --resume    /path/to/phoneme_output/best_phoneme_model.pt
"""

import os
import sys
import yaml
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from dataset import (
    get_train_test_datasets,
    get_suitcase_train_test_datasets,
    collate_fn,
)
from wav2vec2_phonological import PhonemeLevelWav2Vec2
from phonological_features import (
    CMU_39_PHONEMES,
    PHONEME_TO_IDX,
    phoneme_sequence_to_feature_sequences,
    NUM_FEATURES,
)
from alignment import levenshtein_alignment
from metrics import (
    compute_all_feature_metrics,
    print_feature_metrics,
    FeatureMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_PHONEMES = 39
BLANK_IDX    = 39   # CTC blank = last node (index 39)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_attention_mask(
    input_values: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    B, T = input_values.shape
    mask = torch.zeros(B, T, dtype=torch.long, device=input_values.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1
    return mask


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CTC greedy decode → phoneme strings
# ─────────────────────────────────────────────────────────────────────────────

def ctc_decode(logits: torch.Tensor) -> list[list[str]]:
    """
    Greedy CTC decode for the phoneme-level model.

    Args:
        logits: (B, T, 40) — raw logits

    Returns:
        List of B phoneme sequences (as string lists).
    """
    preds = logits.argmax(dim=-1)   # (B, T)
    results = []
    for b in range(preds.shape[0]):
        seq  = []
        prev = -1
        for p in preds[b].tolist():
            if p == BLANK_IDX:
                prev = -1
                continue
            if p != prev:
                seq.append(CMU_39_PHONEMES[p])
                prev = p
        seq = [ph for ph in seq if ph != "sil"]
        results.append(seq)
    return results


def phones_to_feature_seqs(phones: list[str]) -> list[list[bool]]:
    """
    Convert a phoneme sequence to 35 binary feature sequences.

    This is the bridge between the phoneme model output and the
    feature-level evaluation. Produces the same [35][U] bool format
    expected by compute_all_feature_metrics().
    """
    if not phones:
        return [[] for _ in range(NUM_FEATURES)]
    int_seqs = phoneme_sequence_to_feature_sequences(phones)
    return [[bool(v) for v in seq] for seq in int_seqs]


# ─────────────────────────────────────────────────────────────────────────────
# PER metric — for monitoring during training
# ─────────────────────────────────────────────────────────────────────────────

class PERCounts:
    """Accumulates PER = (S+D+I)/N across utterances."""
    def __init__(self):
        self.S = self.D = self.I = self.N = 0

    @property
    def PER(self) -> float:
        return (self.S + self.D + self.I) / self.N if self.N > 0 else 0.0

    def update(self, ref: list[str], hyp: list[str]) -> None:
        s, d, i, _ = levenshtein_alignment(ref, hyp)
        self.S += s; self.D += d; self.I += i; self.N += len(ref)

    def summary(self) -> str:
        return (f"PER={self.PER*100:.2f}%  "
                f"(S={self.S} D={self.D} I={self.I} N={self.N})")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: PhonemeLevelWav2Vec2,
    loss_fn: nn.CTCLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_accum: int,
    epoch: int,
    log_every: int = 50,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_values   = batch["input_values"].to(device)
        input_lengths  = batch["input_lengths"].to(device)
        attention_mask = build_attention_mask(input_values, input_lengths)

        logits, output_lengths = model(input_values, attention_mask)
        logits_t = logits.transpose(0, 1)   # (T, B, 40)
        output_lengths = output_lengths.clamp(max=logits_t.shape[0])

        # CTC targets: actual_phones → phoneme indices (sil excluded)
        targets        = []
        target_lengths = []
        for phones in batch["actual_phones"]:
            ids = [
                PHONEME_TO_IDX[p]
                for p in phones
                if p in PHONEME_TO_IDX and p != "sil"
            ]
            if not ids:
                ids = [0]
            targets.append(torch.tensor(ids, dtype=torch.long))
            target_lengths.append(len(ids))

        targets_padded = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True
        ).to(device)
        target_lengths_t = torch.tensor(
            target_lengths, dtype=torch.long, device=device
        )

        log_probs = F.log_softmax(logits_t, dim=-1)
        loss = loss_fn(log_probs, targets_padded, output_lengths, target_lengths_t)

        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batches  += 1

        if (step + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch} | Step {step+1}/{len(loader)} | "
                f"Loss={total_loss/n_batches:.4f} | LR={lr:.2e}"
            )

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — output format identical to train.py
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: PhonemeLevelWav2Vec2,
    loader: DataLoader,
    device: torch.device,
    label: str = "Test",
) -> tuple[list[FeatureMetrics], PERCounts]:
    """
    Evaluate the phoneme-level model.

    Produces output identical in structure to train.py's evaluate():
      - Returns list[FeatureMetrics] with 35 entries
      - Each entry has .accuracy, .fer, .f1, .precision, .recall
      - Printed by print_feature_metrics() — same function as train.py

    Pipeline:
      audio
        → CTC decode → recognized phoneme strings
        → phones_to_feature_seqs() → [35][U_hyp] bool feature sequences
        → compute_all_feature_metrics(ref_feat_seqs, hyp_feat_seqs)
        → list[FeatureMetrics]

    Reference is built from actual_phones exactly as in train.py.
    """
    model.eval()
    all_ref_seqs = []   # [N_utts][35][U_ref] bool
    all_hyp_seqs = []   # [N_utts][35][U_hyp] bool
    per_counts   = PERCounts()

    for batch_idx, batch in enumerate(loader):
        input_values   = batch["input_values"].to(device)
        input_lengths  = batch["input_lengths"].to(device)
        attention_mask = build_attention_mask(input_values, input_lengths)

        logits, _        = model(input_values, attention_mask)
        recognized_batch = ctc_decode(logits)

        for b in range(len(batch["actual_phones"])):
            actual = [p for p in batch["actual_phones"][b] if p != "sil"]
            recog  = recognized_batch[b]

            # PER accumulation
            if actual:
                per_counts.update(actual, recog)

            # Reference: actual_phones → feature sequences
            ref_feat_seqs = phones_to_feature_seqs(actual)

            # Hypothesis: recognized_phones → feature sequences
            # If model output is empty, produce empty sequences
            hyp_feat_seqs = phones_to_feature_seqs(recog)

            all_ref_seqs.append(ref_feat_seqs)
            all_hyp_seqs.append(hyp_feat_seqs)

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  [{label}] {batch_idx+1}/{len(loader)} batches | "
                f"PER={per_counts.PER*100:.2f}%"
            )

    metrics_list = compute_all_feature_metrics(all_ref_seqs, all_hyp_seqs)
    return metrics_list, per_counts


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train phoneme-level CTC baseline for MDD comparison"
    )
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume",     type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets — identical split to SCTC-SB model ──────────────────────
    train_datasets = []


    l2_train_ds, l2_test_ds = get_train_test_datasets(
        l2arctic_root=args.data_dir,
        split="scripted",
        max_duration=cfg["data"]["max_duration"],
    )
    train_datasets.append(l2_train_ds)
    logger.info(f"L2-Scripted: train={len(l2_train_ds)} | test={len(l2_test_ds)}")

    suit_train_ds, suit_test_ds = get_suitcase_train_test_datasets(
        l2arctic_root=args.data_dir,
        max_chunk_duration=10.0,
    )
    train_datasets.append(suit_train_ds)
    logger.info(f"Suitcase: train={len(suit_train_ds)} | test={len(suit_test_ds)}")

    train_ds = ConcatDataset(train_datasets)
    logger.info(f"Total train utterances: {len(train_ds)}")

    bs          = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        l2_test_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    suit_loader = DataLoader(
        suit_test_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = PhonemeLevelWav2Vec2(
        pretrained_model_name=cfg["model"]["pretrained_model_name"],
        num_phonemes=NUM_PHONEMES,
        freeze_cnn_encoder=cfg["model"]["freeze_cnn_encoder"],
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: total={total:,} | trainable={trainable:,}")

    # ── Loss ─────────────────────────────────────────────────────────────
    loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    # ── Optimizer + Scheduler — identical to train.py ────────────────────
    num_epochs   = cfg["training"]["num_epochs"]
    grad_accum   = cfg["training"]["gradient_accumulation_steps"]
    total_steps  = (len(train_loader) // grad_accum) * num_epochs
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(
        f"Epochs={num_epochs} | Steps={total_steps} | "
        f"Warmup={warmup_steps} | GradAccum={grad_accum}"
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt.get('epoch', '?')}")

    # ── Training loop ─────────────────────────────────────────────────────
    # Saved by Avg ACC (feature-level) — matches train.py criterion exactly
    best_avg_acc = 0.0
    log_every  = cfg["training"].get("log_every", 50)
    save_every = cfg["training"].get("save_every", 5)

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"  EPOCH {epoch}/{num_epochs}")
        logger.info(f"{'='*60}")

        t0         = time.time()
        train_loss = train_epoch(
            model, loss_fn, train_loader,
            optimizer, scheduler,
            device, grad_accum, epoch, log_every,
        )
        elapsed = time.time() - t0
        logger.info(f"  Epoch {epoch} train loss: {train_loss:.4f} | time: {elapsed:.0f}s")

        logger.info("  Evaluating on L2-Scripted test set...")
        metrics_s, per_s = evaluate(model, test_loader, device, "L2-Scripted")
        avg_acc_s = sum(m.accuracy for m in metrics_s) / len(metrics_s)
        avg_f1_s  = sum(m.f1       for m in metrics_s) / len(metrics_s)
        logger.info(
            f"  [Scripted] Avg ACC: {avg_acc_s*100:.2f}% | "
            f"Avg F1: {avg_f1_s*100:.2f}% | {per_s.summary()}"
        )

        logger.info("  Evaluating on L2-Suitcase test set...")
        metrics_u, per_u = evaluate(model, suit_loader, device, "L2-Suitcase")
        avg_acc_u = sum(m.accuracy for m in metrics_u) / len(metrics_u)
        avg_f1_u  = sum(m.f1       for m in metrics_u) / len(metrics_u)
        logger.info(
            f"  [Suitcase] Avg ACC: {avg_acc_u*100:.2f}% | "
            f"Avg F1: {avg_f1_u*100:.2f}% | {per_u.summary()}"
        )

        # Save best by Scripted Avg ACC — matches train.py
        if avg_acc_s > best_avg_acc:
            best_avg_acc = avg_acc_s
            best_path = Path(args.output_dir) / "best_phoneme_model.pt"
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss":                 train_loss,
            }, best_path)
            logger.info(
                f"  ★ New best model saved "
                f"(Scripted ACC={best_avg_acc*100:.2f}%)"
            )

        if epoch % save_every == 0:
            ckpt_path = Path(args.output_dir) / f"phoneme_epoch_{epoch:02d}.pt"
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss":                 train_loss,
            }, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

    # ── Final evaluation — output format identical to train.py ────────────
    logger.info("\nFinal evaluation with best model...")
    ckpt = torch.load(
        Path(args.output_dir) / "best_phoneme_model.pt",
        map_location=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    logger.info("\n--- L2-Scripted ---")
    metrics_s, per_s = evaluate(model, test_loader, device, "L2-Scripted")
    logger.info(f"PER: {per_s.summary()}")
    print_feature_metrics(metrics_s)

    logger.info("\n--- L2-Suitcase ---")
    metrics_u, per_u = evaluate(model, suit_loader, device, "L2-Suitcase")
    logger.info(f"PER: {per_u.summary()}")
    print_feature_metrics(metrics_u)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
