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

Evaluation mirrors train.py exactly: scripted test set and suitcase test set
are evaluated separately at every epoch. Model selection uses Scripted PER
(lower is better).

Epoch metrics  — PER on L2-Scripted and L2-Suitcase
Final report   — per-phoneme breakdown (precision, recall, F1, support)
                 printed for both test sets after training completes.

=============================================================================

Usage:
    python train_phoneme.py \
        --config    config.yaml \
        --data_dir  /path/to/l2arctic \
        --output_dir /path/to/phoneme_output

    # With TIMIT (recommended):
    python train_phoneme.py \
        --config    config.yaml \
        --data_dir  /path/to/l2arctic \
        --timit_dir /path/to/timit \
        --output_dir /path/to/phoneme_output

    # Resume:
    python train_phoneme.py \
        --config     config.yaml \
        --data_dir   /path/to/l2arctic \
        --output_dir /path/to/phoneme_output \
        --resume     /path/to/phoneme_output/checkpoint_phoneme_epoch_05.pt
"""

import os
import sys
import yaml
import argparse
import logging
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, Wav2Vec2FeatureExtractor

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_datasets_separate, make_collate_fn
from wav2vec2_phonological import PhonemeLevelWav2Vec2
from phonological_features import CMU_39_PHONEMES, PHONEME_TO_IDX
from alignment import levenshtein_alignment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

NUM_PHONEMES = 39
BLANK_IDX    = 39


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def ctc_decode(logits: torch.Tensor) -> list:
    preds = logits.argmax(dim=-1)
    results = []
    for b in range(preds.shape[0]):
        seq, prev = [], -1
        for p in preds[b].tolist():
            if p == BLANK_IDX:
                prev = -1
                continue
            if p != prev:
                seq.append(CMU_39_PHONEMES[p])
                prev = p
        results.append([ph for ph in seq if ph != "sil"])
    return results


class PERCounts:
    def __init__(self):
        self.S = self.D = self.I = self.N = 0

    @property
    def PER(self) -> float:
        return (self.S + self.D + self.I) / self.N if self.N > 0 else 0.0

    def update(self, ref: list, hyp: list) -> None:
        s, d, i, _ = levenshtein_alignment(ref, hyp)
        self.S += s; self.D += d; self.I += i; self.N += len(ref)

    def summary(self) -> str:
        return (f"PER={self.PER*100:.2f}%  "
                f"(S={self.S} D={self.D} I={self.I} N={self.N})")


class PhonemeStats:
    """Accumulates TP/FP/FN per phoneme for a final recognition report."""

    def __init__(self):
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

    def update_from_alignment(self, ref: list, hyp: list) -> None:
        _, _, _, aligned_pairs = levenshtein_alignment(ref, hyp)
        for r, h in aligned_pairs:
            if r is not None and h is not None:
                if r == h:
                    self.tp[r] += 1
                else:
                    self.fn[r] += 1   # substitution — ref missed
                    self.fp[h] += 1   # substitution — wrong hyp output
            elif r is not None:
                self.fn[r] += 1       # deletion
            elif h is not None:
                self.fp[h] += 1       # insertion

    def print_report(self, label: str) -> None:
        cmu_set = set(CMU_39_PHONEMES)
        all_phonemes = sorted(
            p for p in (set(self.tp) | set(self.fn) | set(self.fp))
            if p in cmu_set
        )

        logger.info(f"\n  [{label}] Per-phoneme recognition metrics:")
        header = f"  {'Phoneme':<10} {'PRE%':>7} {'REC%':>7} {'F1%':>7} {'Support':>9}"
        sep    = "  " + "-" * (len(header) - 2)
        logger.info(header)
        logger.info(sep)

        pre_vals, rec_vals, f1_vals = [], [], []
        total_tp = total_fp = total_fn = 0

        for ph in all_phonemes:
            tp = self.tp[ph]; fp = self.fp[ph]; fn = self.fn[ph]
            pre = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
            f1  = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
            logger.info(f"  {ph:<10} {pre:>7.2f} {rec:>7.2f} {f1:>7.2f} {tp+fn:>9d}")
            pre_vals.append(pre); rec_vals.append(rec); f1_vals.append(f1)
            total_tp += tp; total_fp += fp; total_fn += fn

        macro_pre = sum(pre_vals) / len(pre_vals) if pre_vals else 0.0
        macro_rec = sum(rec_vals) / len(rec_vals) if rec_vals else 0.0
        macro_f1  = sum(f1_vals)  / len(f1_vals)  if f1_vals  else 0.0

        n_ref     = total_tp + total_fn
        micro_pre = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0.0
        micro_f1  = (2 * micro_pre * micro_rec / (micro_pre + micro_rec)
                     if (micro_pre + micro_rec) > 0 else 0.0)

        logger.info(sep)
        logger.info(f"  {'MACRO AVG':<10} {macro_pre:>7.2f} {macro_rec:>7.2f} {macro_f1:>7.2f} {n_ref:>9d}")
        logger.info(f"  {'MICRO AVG':<10} {micro_pre:>7.2f} {micro_rec:>7.2f} {micro_f1:>7.2f} {n_ref:>9d}")


def train_epoch(model, loss_fn, loader, optimizer, scheduler,
                device, grad_accum, epoch, log_every=50) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_values   = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits, output_lengths = model(input_values, attention_mask)
        logits_t       = logits.transpose(0, 1)   # (T, B, 40)
        output_lengths = output_lengths.clamp(max=logits_t.shape[0])

        targets, target_lengths = [], []
        for phones in batch["actual_phones"]:
            ids = [PHONEME_TO_IDX[p] for p in phones
                   if p in PHONEME_TO_IDX and p != "sil"]
            if not ids:
                ids = [0]
            targets.append(torch.tensor(ids, dtype=torch.long))
            target_lengths.append(len(ids))

        targets_padded   = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)
        target_lengths_t = torch.tensor(target_lengths, dtype=torch.long, device=device)

        log_probs = F.log_softmax(logits_t, dim=-1)
        loss      = loss_fn(log_probs, targets_padded, output_lengths, target_lengths_t)
        (loss / grad_accum).backward()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batches  += 1

        if (step + 1) % log_every == 0:
            logger.info(
                f"  Epoch {epoch} | Step {step+1}/{len(loader)} | "
                f"Loss={total_loss/n_batches:.4f} | LR={optimizer.param_groups[0]['lr']:.2e}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, label="Test",
             collect_phoneme_stats=False):
    model.eval()
    per_counts    = PERCounts()
    phoneme_stats = PhonemeStats() if collect_phoneme_stats else None

    total_ref_len = 0
    total_hyp_len = 0
    n_utts        = 0

    for batch_idx, batch in enumerate(loader):
        input_values   = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits, _      = model(input_values, attention_mask)
        recognized     = ctc_decode(logits)

        for b in range(len(batch["actual_phones"])):
            actual = [p for p in batch["actual_phones"][b] if p != "sil"]
            recog  = recognized[b]
            total_ref_len += len(actual)
            total_hyp_len += len(recog)
            n_utts        += 1
            if actual:
                per_counts.update(actual, recog)
                if phoneme_stats is not None:
                    phoneme_stats.update_from_alignment(actual, recog)

        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"  [{label}] {batch_idx+1}/{len(loader)} batches | "
                f"PER={per_counts.PER*100:.2f}%"
            )

    avg_ref = total_ref_len / max(n_utts, 1)
    avg_hyp = total_hyp_len / max(n_utts, 1)
    logger.info(f"  [{label}] avg_ref_len={avg_ref:.1f} | avg_hyp_len={avg_hyp:.1f}")

    return per_counts, phoneme_stats


def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir):
    path = Path(output_dir) / f"checkpoint_phoneme_epoch_{epoch:02d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss":                 loss,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train phoneme-level CTC baseline for MDD comparison"
    )
    parser.add_argument("--config",     type=str, default="config.yaml")
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--timit_dir",  type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume",     type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    pretrained_name = cfg["model"]["pretrained_model_name"]
    logger.info(f"Loading feature extractor from '{pretrained_name}' ...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_name)
    logger.info("Feature extractor loaded.")

    # ── Datasets — same separate scripted/suitcase split as train.py ──────
    train_ds, scripted_test_ds, suitcase_test_ds = get_datasets_separate(
        l2arctic_root=args.data_dir,
        timit_root=args.timit_dir,
        max_duration=cfg["data"]["max_duration"],
        max_chunk_duration=10.0,
    )

    bs          = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)
    collate_fn  = make_collate_fn(feature_extractor)

    train_loader    = DataLoader(train_ds,          batch_size=bs, shuffle=True,
                                 collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    scripted_loader = DataLoader(scripted_test_ds,  batch_size=bs, shuffle=False,
                                 collate_fn=collate_fn, num_workers=num_workers)
    suitcase_loader = DataLoader(suitcase_test_ds,  batch_size=bs, shuffle=False,
                                 collate_fn=collate_fn, num_workers=num_workers)

    model = PhonemeLevelWav2Vec2(
        pretrained_model_name=pretrained_name,
        num_phonemes=NUM_PHONEMES,
        freeze_cnn_encoder=cfg["model"]["freeze_cnn_encoder"],
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: total={total:,} | trainable={trainable:,}")

    loss_fn      = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)
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
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    logger.info(
        f"Epochs={num_epochs} | Steps={total_steps} | "
        f"Warmup={warmup_steps} | GradAccum={grad_accum}"
    )

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
    # Model selection: lower Scripted PER is better
    best_scripted_per = float("inf")
    log_every         = cfg["training"].get("log_every", 50)
    save_every        = cfg["training"].get("save_every", 5)

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"  EPOCH {epoch}/{num_epochs}")
        logger.info(f"{'='*60}")

        t0         = time.time()
        train_loss = train_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            device, grad_accum, epoch, log_every,
        )
        logger.info(f"  Epoch {epoch} loss={train_loss:.4f} | time={time.time()-t0:.0f}s")

        logger.info("  Evaluating on L2-Scripted...")
        sc_per, _ = evaluate(model, scripted_loader, device, "L2-Scripted")
        logger.info(f"  [L2-Scripted] {sc_per.summary()}")

        logger.info("  Evaluating on L2-Suitcase...")
        su_per, _ = evaluate(model, suitcase_loader, device, "L2-Suitcase")
        logger.info(f"  [L2-Suitcase] {su_per.summary()}")

        if sc_per.PER < best_scripted_per:
            best_scripted_per = sc_per.PER
            torch.save({
                "epoch": epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss":                 train_loss,
            }, Path(args.output_dir) / "best_phoneme_model.pt")
            logger.info(f"  ★ New best model (Scripted PER={best_scripted_per*100:.2f}%)")
        else:
            logger.info(
                f"  No improvement this epoch "
                f"(best Scripted PER={best_scripted_per*100:.2f}%)"
            )

        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, args.output_dir)

    # ── Final evaluation: best model + full per-phoneme breakdown ──────────
    logger.info("\nFinal evaluation with best model...")
    ckpt = torch.load(Path(args.output_dir) / "best_phoneme_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded best model from epoch {ckpt.get('epoch', '?')}")

    logger.info("\n=== L2-SCRIPTED (Final) ===")
    sc_per, sc_stats = evaluate(model, scripted_loader, device, "L2-Scripted",
                                collect_phoneme_stats=True)
    logger.info(f"  [L2-Scripted] {sc_per.summary()}")
    sc_stats.print_report("L2-Scripted")

    logger.info("\n=== L2-SUITCASE (Final) ===")
    su_per, su_stats = evaluate(model, suitcase_loader, device, "L2-Suitcase",
                                collect_phoneme_stats=True)
    logger.info(f"  [L2-Suitcase] {su_per.summary()}")
    su_stats.print_report("L2-Suitcase")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
