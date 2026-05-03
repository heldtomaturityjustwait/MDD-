"""
train.py
========
Training entry point for phonological-level wav2vec2 MDD.
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

from dataset import L2ArcticDataset, collate_fn, get_train_test_datasets, get_suitcase_train_test_datasets
from phonological_features import (
    phoneme_sequence_to_feature_sequences,
    PHONOLOGICAL_FEATURES,
    NUM_FEATURES,
)
from wav2vec2_phonological import PhonologicalWav2Vec2
from sctc_loss import SCTCSBLoss
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


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_attention_mask(input_values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, T = input_values.shape
    mask = torch.zeros(B, T, dtype=torch.long, device=input_values.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1
    return mask


def train_epoch(
    model, loss_fn, loader, optimizer, scheduler,
    device, grad_accum_steps, epoch, log_every=50,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_values = batch["input_values"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        ctc_labels = batch["ctc_labels"]

        attention_mask = build_attention_mask(input_values, input_lengths)
        logits, output_lengths = model(input_values, attention_mask)
        logits_t = logits.transpose(0, 1)
        output_lengths = output_lengths.clamp(max=logits_t.shape[0])

        loss = loss_fn(logits_t, ctc_labels, output_lengths)
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        n_batches += 1

        if (step + 1) % log_every == 0:
            avg = total_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            logger.info(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | "
                        f"Loss={avg:.4f} | LR={lr:.2e}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> list:
    """
    Evaluate phonological feature recognition accuracy.
    Reference = actual_phones (what speaker actually said, from annotations).
    Measures how well the model recognizes actual pronounced features.
    """
    model.eval()
    all_ref_seqs = []
    all_hyp_seqs = []

    for batch in loader:
        input_values = batch["input_values"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        attention_mask = build_attention_mask(input_values, input_lengths)

        logits, _ = model(input_values, attention_mask)
        decoded_batch = model.decode(logits)

        for b, phones in enumerate(batch["actual_phones"]):
            ref_feature_seqs = phoneme_sequence_to_feature_sequences(phones)
            ref_bool = [[bool(v) for v in seq] for seq in ref_feature_seqs]
            all_ref_seqs.append(ref_bool)
            all_hyp_seqs.append(decoded_batch[b])

    return compute_all_feature_metrics(all_ref_seqs, all_hyp_seqs)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, output_dir):
    ckpt_path = Path(output_dir) / f"checkpoint_epoch_{epoch:02d}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="scripted",
                        choices=["scripted", "suitcase"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["paths"]["l2arctic_dir"] = args.data_dir
    if args.output_dir:
        cfg["paths"]["output_dir"] = args.output_dir

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets ──────────────────────────────────────────────────────────
    from dataset import get_train_test_datasets, get_suitcase_train_test_datasets, collate_fn
    from torch.utils.data import ConcatDataset
    
    train_ds, test_ds = get_train_test_datasets(
        l2arctic_root=cfg["paths"]["l2arctic_dir"],
        split=args.split,
        max_duration=cfg["data"]["max_duration"],
    )

    # ── Suitcase train + test sets ─────────────────────────────────────────
    # Suitcase: 22 speakers (ASI+SKA excluded), same 6 test speakers as scripted
    # Train: 16 suitcase speakers, Test: 6 suitcase speakers
    suitcase_train_ds, suitcase_test_ds = get_suitcase_train_test_datasets(
        l2arctic_root=cfg["paths"]["l2arctic_dir"],
        max_chunk_duration=10.0,
    )

    # Add suitcase training data to combined training set
    train_ds = ConcatDataset([train_ds, suitcase_train_ds])
    logger.info(f"Added suitcase train ({len(suitcase_train_ds)} chunks). "
                f"Total train: {len(train_ds)}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    suitcase_loader = DataLoader(
        suitcase_test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    # ── Model ─────────────────────────────────────────────────────────────
    model = PhonologicalWav2Vec2(
        pretrained_model_name=cfg["model"]["pretrained_model_name"],
        num_output_nodes=cfg["model"]["num_output_nodes"],
        freeze_cnn_encoder=cfg["model"]["freeze_cnn_encoder"],
    ).to(device)

    param_info = model.count_parameters()
    logger.info(f"Parameters: total={param_info['total']:,} | "
                f"trainable={param_info['trainable']:,} | "
                f"frozen={param_info['frozen']:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────
    loss_fn = SCTCSBLoss(reduction="mean")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs = cfg["training"]["num_epochs"]
    grad_accum = cfg["training"]["gradient_accumulation_steps"]
    total_steps = (len(train_loader) // grad_accum) * num_epochs
    warmup_steps = int(total_steps * cfg["training"]["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

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

        t0 = time.time()
        train_loss = train_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            device, grad_accum, epoch,
            log_every=cfg["training"]["log_every"],
        )
        elapsed = time.time() - t0
        logger.info(f"  Epoch {epoch} train loss: {train_loss:.4f} | time: {elapsed:.0f}s")

        logger.info(f"  Evaluating on L2-Scripted test set...")
        metrics_list = evaluate(model, test_loader, device)
        avg_acc = sum(m.accuracy for m in metrics_list) / len(metrics_list)
        avg_f1  = sum(m.f1       for m in metrics_list) / len(metrics_list)
        logger.info(f"  [Scripted] Avg ACC: {avg_acc*100:.2f}% | Avg F1: {avg_f1*100:.2f}%")

        logger.info(f"  Evaluating on L2-Suitcase test set...")
        suit_metrics = evaluate(model, suitcase_loader, device)
        suit_acc = sum(m.accuracy for m in suit_metrics) / len(suit_metrics)
        suit_f1  = sum(m.f1       for m in suit_metrics) / len(suit_metrics)
        logger.info(f"  [Suitcase] Avg ACC: {suit_acc*100:.2f}% | Avg F1: {suit_f1*100:.2f}%")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_path = Path(output_dir) / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            logger.info(f"  ★ New best model saved (Scripted ACC={avg_acc*100:.2f}%)")

        if epoch % cfg["training"]["save_every"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, output_dir)

    logger.info("\nFinal evaluation with best model...")
    model.load_state_dict(torch.load(Path(output_dir) / "best_model.pt"))

    logger.info("\n--- L2-Scripted ---")
    metrics_list = evaluate(model, test_loader, device)
    print_feature_metrics(metrics_list)

    logger.info("\n--- L2-Suitcase ---")
    suit_metrics = evaluate(model, suitcase_loader, device)
    print_feature_metrics(suit_metrics)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()