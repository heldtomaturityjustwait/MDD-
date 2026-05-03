"""
train.py
========
Training entry point for phonological-feature wav2vec2 MDD.

Data sources (all optional except L2-ARCTIC annotated):
  --timit_dir   : adds TIMIT TRAIN split (recommended for better coverage)
  --data_dir    : L2-ARCTIC root (annotated utterances + suitcase)

Usage:
    python train.py --config config.yaml \
        --data_dir  /path/to/l2arctic \
        --timit_dir /path/to/timit      # optional but recommended
        --output_dir /path/to/output

    # Resume:
    python train.py --config config.yaml \
        --data_dir  /path/to/l2arctic \
        --output_dir /path/to/output \
        --resume    /path/to/output/checkpoint_epoch_05.pt
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
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_datasets, collate_fn
from phonological_features import (
    phoneme_sequence_to_feature_sequences,
    NUM_FEATURES,
)
from wav2vec2_phonological import PhonologicalWav2Vec2
from sctc_loss import SCTCSBLoss
from metrics import compute_all_feature_metrics, print_feature_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_attention_mask(
    input_values: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    B, T = input_values.shape
    mask = torch.zeros(B, T, dtype=torch.long, device=input_values.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model, loss_fn, loader, optimizer, scheduler,
    device, grad_accum_steps, epoch, log_every=50,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_values  = batch["input_values"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        ctc_labels    = batch["ctc_labels"]

        attention_mask = build_attention_mask(input_values, input_lengths)
        logits, output_lengths = model(input_values, attention_mask)
        logits_t       = logits.transpose(0, 1)           # (T, B, 71)
        output_lengths = output_lengths.clamp(max=logits_t.shape[0])

        loss = loss_fn(logits_t, ctc_labels, output_lengths) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        n_batches  += 1

        if (step + 1) % log_every == 0:
            avg = total_loss / n_batches
            lr  = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch} | Step {step+1}/{len(loader)} | "
                f"Loss={avg:.4f} | LR={lr:.2e}"
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> list:
    """
    Evaluate phonological feature recognition.
    Reference = actual_phones from annotations.
    """
    model.eval()
    all_ref, all_hyp = [], []

    for batch in loader:
        input_values  = batch["input_values"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        attention_mask = build_attention_mask(input_values, input_lengths)

        logits, _ = model(input_values, attention_mask)
        decoded   = model.decode(logits)    # [B][35][list[bool]]

        for b, phones in enumerate(batch["actual_phones"]):
            ref_feat = phoneme_sequence_to_feature_sequences(phones)
            ref_bool = [[bool(v) for v in seq] for seq in ref_feat]
            all_ref.append(ref_bool)
            all_hyp.append(decoded[b])

    return compute_all_feature_metrics(all_ref, all_hyp)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir):
    path = Path(output_dir) / f"checkpoint_epoch_{epoch:02d}.pt"
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss":                 loss,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, default="config.yaml")
    parser.add_argument("--data_dir",    type=str, required=True,
                        help="L2-ARCTIC root directory")
    parser.add_argument("--timit_dir",   type=str, default=None,
                        help="TIMIT root directory (optional, adds TRAIN split)")
    parser.add_argument("--output_dir",  type=str, default=None)
    parser.add_argument("--resume",      type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Build datasets ────────────────────────────────────────────────────
    train_ds, test_ds = get_datasets(
        l2arctic_root=args.data_dir,
        timit_root=args.timit_dir,
        max_duration=cfg["data"]["max_duration"],
        max_chunk_duration=10.0,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    bs          = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = PhonologicalWav2Vec2(
        pretrained_model_name=cfg["model"]["pretrained_model_name"],
        num_output_nodes=cfg["model"]["num_output_nodes"],
        freeze_cnn_encoder=cfg["model"]["freeze_cnn_encoder"],
    ).to(device)

    info = model.count_parameters()
    logger.info(f"Parameters: total={info['total']:,} | "
                f"trainable={info['trainable']:,} | frozen={info['frozen']:,}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────────
    loss_fn = SCTCSBLoss(reduction="mean")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs   = cfg["training"]["num_epochs"]
    grad_accum   = cfg["training"]["gradient_accumulation_steps"]
    total_steps  = (len(train_loader) // grad_accum) * num_epochs
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(f"Total steps: {total_steps} | Warmup: {warmup_steps}")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']}")

    # ── Training loop ─────────────────────────────────────────────────────
    best_avg_acc = 0.0

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"  EPOCH {epoch}/{num_epochs}")
        logger.info(f"{'='*60}")

        t0         = time.time()
        train_loss = train_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            device, grad_accum, epoch, cfg["training"]["log_every"],
        )
        logger.info(
            f"  Epoch {epoch} loss={train_loss:.4f} | "
            f"time={time.time()-t0:.0f}s"
        )

        logger.info("  Evaluating on test set...")
        metrics = evaluate(model, test_loader, device)
        acc = sum(m.accuracy for m in metrics) / len(metrics)
        f1  = sum(m.f1       for m in metrics) / len(metrics)
        logger.info(f"  Avg ACC={acc*100:.2f}% | F1={f1*100:.2f}%")

        if acc > best_avg_acc:
            best_avg_acc = acc
            best_path = Path(output_dir) / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            logger.info(f"  ★ New best model (ACC={acc*100:.2f}%)")

        if epoch % cfg["training"]["save_every"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, output_dir)

    # ── Final evaluation ──────────────────────────────────────────────────
    logger.info("\nFinal evaluation with best model...")
    model.load_state_dict(torch.load(Path(output_dir) / "best_model.pt"))
    print_feature_metrics(evaluate(model, test_loader, device))

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
