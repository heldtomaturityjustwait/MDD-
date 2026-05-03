"""
dataset.py
==========
PyTorch Dataset for the full L2-ARCTIC corpus.

L2-ARCTIC structure:
    l2arctic/
      <SPEAKER_ID>/
        wav/          ← .wav files (16kHz)
        annotation/   ← .TextGrid files (human phoneme-level annotation with
                        error labels: C/S/I/D). Only ~15% of scripted utterances
                        are annotated. NO .txt files exist — TextGrid only.
        textgrid/     ← .TextGrid files for ALL utterances. These are
                        force-alignment-derived canonical phoneme transcriptions
                        (machine-generated, no error labels). Used as the
                        authoritative canonical reference and as the fallback
                        actual_phones source for unannotated utterances.
      suitcase_corpus/
        wav/          ← one .wav per speaker (spontaneous speech)
        annotation/   ← one .TextGrid per speaker (fully annotated with C/S/I/D)

Training strategy (following Shahin et al. 2025):
  - ALL L2-ARCTIC scripted utterances are used for training.
    • Annotated (~15%): actual_phones derived from error labels in annotation/
      TextGrid; canonical_phones from textgrid/ forced-alignment.
    • Unannotated (~85%): textgrid/ forced-alignment phones used as both
      actual and canonical (no errors assumed for unlabelled utterances).
  - Suitcase corpus used for evaluation only.
  - No TIMIT or LibriSpeech — L2-ARCTIC alone is the training source.
"""

import os
import re
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

from phonological_features import (
    PHONEME_TO_IDX,
    CMU_39_PHONEMES,
    phoneme_sequence_to_feature_sequences,
    feature_sequences_to_ctc_labels,
    NUM_FEATURES,
)

# ─────────────────────────────────────────────────────────────────────────────
# L2-ARCTIC speaker split (paper: Ye et al. 2022 split)
# One speaker per L1 language forms the test set
# ─────────────────────────────────────────────────────────────────────────────
SCRIPTED_TEST_SPEAKERS = {"ASI", "YBAA", "HJK", "BWC", "EBVS", "YDCK"}  # Ye et al. L2-Scripted split used in the paper
SUITCASE_TEST_SPEAKERS = {"RRBI", "YBAA", "HJK", "BWC", "EBVS", "YDCK"}  # ASI has no suitcase recording


def normalize_phoneme(ph: str) -> str:
    """Strip stress markers and lowercase. e.g. 'AH1' -> 'ah'."""
    ph = ph.lower().strip()
    ph = re.sub(r"[0-9]", "", ph)
    ph = ph.replace("ax", "ah")
    ph = ph.replace("ix", "ih")
    ph = ph.replace("nx", "n")
    ph = ph.replace("em", "m")
    ph = ph.replace("en", "n")
    ph = ph.replace("eng", "ng")
    if ph not in CMU_39_PHONEMES:
        return "sil"
    return ph


def parse_textgrid_canonical(textgrid_path: str) -> list[str]:
    """
    Extract clean canonical phoneme sequence from an L2-Arctic scripted
    TextGrid file (textgrid/ folder, NOT annotation/ folder).

    These TextGrid files contain the reference pronunciation with no error
    labels -- just clean ARPAbet phonemes like AO1, TH, ER0.
    Authoritative source for canonical_phones in L2-Scripted.
    """
    with open(textgrid_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    tier_blocks = re.split(r'item\s*\[\d+\]', content)
    phones_block = None
    for block in tier_blocks:
        if re.search(r'name\s*=\s*"phones?"', block, re.IGNORECASE):
            phones_block = block
            break

    if phones_block is None:
        return []

    intervals = re.findall(
        r'intervals\s*\[\d+\].*?xmin\s*=\s*[\d.]+.*?xmax\s*=\s*[\d.]+.*?text\s*=\s*"([^"]*)"',
        phones_block, re.DOTALL
    )

    canonical_phones = []
    for text in intervals:
        text = text.strip()
        if text in ("", "sil", "sp", "spn", "<unk>"):
            continue
        # Take first comma-separated part (no error labels in these files)
        ph_raw = text.split(",")[0].strip()
        ph = normalize_phoneme(ph_raw)
        if ph != "sil":
            canonical_phones.append(ph)

    return canonical_phones


def text_to_phones(text: str) -> list[str]:
    """
    Convert a raw transcript sentence to canonical phoneme sequence
    using CMUdict via the `pronouncing` library.

    Used for L2-Suitcase where only a plain-text transcript exists.
    Words not in CMUdict are silently skipped.
    First pronunciation variant is always used.
    """
    try:
        import pronouncing
    except ImportError:
        raise ImportError(
            "Install pronouncing: pip install pronouncing"
        )

    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    words = text.split()

    phones = []
    for word in words:
        pronunciations = pronouncing.phones_for_word(word)
        if not pronunciations:
            continue
        for ph_raw in pronunciations[0].split():
            ph = normalize_phoneme(ph_raw)
            if ph != "sil":
                phones.append(ph)
    return phones


def parse_textgrid_annotation(textgrid_path: str) -> tuple[list[str], list[str], list[dict]]:
    """
    Parse L2-Arctic annotation TextGrid file.

    L2-Arctic phones tier format (from README):
        Correct:      "AH1"        → canonical=ah, status=C
        Substitution: "DH,D,s"     → canonical=dh, status=S, pronounced=d
        Deletion:     "TH,sil,d"   → canonical=th, status=D
        Addition:     "sil,AH,a"   → status=I, pronounced=ah
        Allophone:    "T,T*,s"     → canonical=t, status=S, pronounced=t
        Hard to judge:"CPL,err,s"  → canonical=cpl, status=S, pronounced=None

    Returns:
        actual_phones    : what the speaker actually said (for CTC training)
        canonical_phones : what should have been said (for MDD evaluation)
        mdd_records      : full annotation records
    """
    with open(textgrid_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    tier_blocks = re.split(r'item\s*\[\d+\]', content)
    phones_block = None
    for block in tier_blocks:
        if re.search(r'name\s*=\s*"phones?"', block, re.IGNORECASE):
            phones_block = block
            break

    if phones_block is None:
        return [], [], []

    intervals = re.findall(
        r'intervals\s*\[\d+\].*?xmin\s*=\s*([\d.]+).*?xmax\s*=\s*([\d.]+).*?text\s*=\s*"([^"]*)"',
        phones_block, re.DOTALL
    )

    actual_phones    = []
    canonical_phones = []
    mdd_records      = []

    for _, _, text in intervals:
        text = text.strip()

        # Skip silence and empty
        if text in ("", "sil", "sp", "spn", "<unk>"):
            continue

        parts = [p.strip() for p in text.split(",")]

        if len(parts) == 1:
            # ── Correct pronunciation ─────────────────────────────────
            ph = normalize_phoneme(parts[0])
            if ph == "sil":
                continue
            actual_phones.append(ph)
            canonical_phones.append(ph)
            mdd_records.append({
                "canonical": ph,
                "status": "C",
                "pronounced": None,
            })

        elif len(parts) == 3:
            canonical_raw, pronounced_raw, error_type = parts
            error_type = error_type.strip().lower()

            if error_type == "s":
                # ── Substitution ──────────────────────────────────────
                canonical = normalize_phoneme(canonical_raw)
                if canonical == "sil":
                    continue

                pronounced_clean = pronounced_raw.replace("*", "").strip()
                if pronounced_clean.lower() == "err":
                    pronounced = None
                else:
                    pronounced = normalize_phoneme(pronounced_clean)
                    if pronounced == "sil":
                        pronounced = None

                # CTC trains on what was actually said
                actual_phones.append(pronounced if pronounced else canonical)
                canonical_phones.append(canonical)
                mdd_records.append({
                    "canonical": canonical,
                    "status": "S",
                    "pronounced": pronounced,
                })

            elif error_type == "d":
                # ── Deletion ──────────────────────────────────────────
                # Speaker said nothing — skip actual_phones
                canonical = normalize_phoneme(canonical_raw)
                if canonical == "sil":
                    continue
                canonical_phones.append(canonical)
                mdd_records.append({
                    "canonical": canonical,
                    "status": "D",
                    "pronounced": None,
                })

            elif error_type == "a":
                # ── Addition (insertion) ──────────────────────────────
                # Extra phoneme said — add to actual, not canonical
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                pronounced = normalize_phoneme(pronounced_clean)
                if pronounced and pronounced != "sil":
                    actual_phones.append(pronounced)
                mdd_records.append({
                    "canonical": None,
                    "status": "I",
                    "pronounced": pronounced if pronounced != "sil" else None,
                })

    return actual_phones, canonical_phones, mdd_records


def parse_l2arctic_annotation(annotation_path: str) -> list[dict]:
    """
    DEAD CODE — L2-ARCTIC does NOT ship .txt annotation files.
    All annotations are in TextGrid format (annotation/<utt_id>.TextGrid).
    This function is kept for API compatibility but should never be called.
    Use parse_textgrid_annotation() instead.

    If this function is ever called it means a stale .txt file was accidentally
    placed in annotation/ — raise loudly so the caller can investigate.
    """
    raise RuntimeError(
        f"parse_l2arctic_annotation() called with path '{annotation_path}', "
        f"but L2-ARCTIC has no .txt annotation files. "
        f"All annotations are TextGrid format. "
        f"Call parse_textgrid_annotation() instead."
    )


class L2ArcticDataset(Dataset):
    """
    Dataset class for L2-ARCTIC corpus.

    Each item returns:
        waveform     : (T,)  float32 tensor at 16kHz
        ctc_labels   : list of 35 lists of int  (one per phonological feature)
        canonical_ph : list of str              (canonical phoneme sequence)
        speaker_id   : str
        utt_id       : str
    """

    def __init__(
        self,
        l2arctic_root: str,
        speakers: list[str],
        split: str = "scripted",      # "scripted" or "suitcase"
        max_duration: float = 15.0,
        sample_rate: int = 16000,
    ):
        self.root = Path(l2arctic_root)
        self.speakers = speakers
        self.split = split
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)

        self.samples = self._collect_samples()
        print(f"[L2ArcticDataset] split={split}, speakers={len(speakers)}, "
              f"utterances={len(self.samples)}")

    def _collect_samples(self) -> list[dict]:
        samples = []
        n_annotated = 0
        n_textgrid_only = 0
        n_skipped = 0

        for spk in self.speakers:
            spk_dir = self.root / spk

            if self.split == "scripted":
                wav_dir = spk_dir / "wav"
                ann_dir = spk_dir / "annotation"
            else:  # suitcase / spontaneous
                wav_dir = spk_dir / "suitcase_corpus" / "wav"
                ann_dir = spk_dir / "suitcase_corpus" / "annotation"

            if not wav_dir.exists():
                # Try flat structure some releases use
                wav_dir = spk_dir
                ann_dir = spk_dir / "annotation"

            if not wav_dir.exists():
                print(f"  [warn] wav dir not found for {spk}: {wav_dir}")
                continue

            tg_canon_dir = spk_dir / "textgrid"

            for wav_file in sorted(wav_dir.glob("*.wav")):
                utt_id = wav_file.stem
                # L2-ARCTIC annotation is always TextGrid — no .txt files exist
                ann_tg_file   = ann_dir      / f"{utt_id}.TextGrid"
                canon_tg_file = tg_canon_dir / f"{utt_id}.TextGrid"

                has_annotation = ann_tg_file.exists()
                has_canon_tg   = canon_tg_file.exists()

                # Skip only if we have absolutely no phoneme source
                if not has_annotation and not has_canon_tg:
                    n_skipped += 1
                    continue

                if has_annotation:
                    n_annotated += 1
                else:
                    # Unannotated utterance — textgrid/ is the only source.
                    # We treat all phonemes as correctly pronounced (status=C).
                    n_textgrid_only += 1

                samples.append({
                    "wav_path":        str(wav_file),
                    "ann_path":        None,                          # no .txt files in L2-ARCTIC
                    "tg_path":         str(ann_tg_file) if has_annotation else None,
                    "canon_tg_path":   str(canon_tg_file) if has_canon_tg else None,
                    "annotation_only": False,
                    "canon_only":      not has_annotation,
                    "speaker":         spk,
                    "utt_id":          utt_id,
                })

        print(f"  [L2ArcticDataset] annotated={n_annotated}, "
              f"textgrid-only={n_textgrid_only}, skipped(no source)={n_skipped}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        # ── Load audio ────────────────────────────────────────────────────
        waveform, sr = torchaudio.load(item["wav_path"])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)  # mono
        # Truncate
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]

        # ── Get phoneme sequences ─────────────────────────────────────────
        actual_phones    = []   # what speaker actually said → CTC training target
        canonical_phones = []   # what should be said → MDD evaluation reference
        mdd_records      = []

        if item.get("canon_only"):
            # ── No annotation: use textgrid/ canonical phones only ────────
            # Treat all phonemes as correctly pronounced (status=C).
            # Covers the ~85% of L2-ARCTIC scripted that has no manual annotation.
            canonical_phones = parse_textgrid_canonical(item["canon_tg_path"])
            actual_phones    = list(canonical_phones)
            mdd_records      = [
                {"canonical": ph, "status": "C", "pronounced": None}
                for ph in canonical_phones
            ]

        else:
            # ── Annotated utterance: annotation/ TextGrid is ground truth ──
            # L2-ARCTIC has no .txt annotation files; all annotations are
            # in TextGrid format in annotation/<utt_id>.TextGrid.
            if item["tg_path"] and os.path.exists(item["tg_path"]):
                actual_phones, _, mdd_records = parse_textgrid_annotation(
                    item["tg_path"]
                )
            # else: should not happen — _collect_samples guarantees tg_path
            # is set whenever canon_only=False

            # ── Canonical phones: prefer textgrid/ clean reference ────────
            if item.get("canon_tg_path") and os.path.exists(item["canon_tg_path"]):
                canonical_phones = parse_textgrid_canonical(item["canon_tg_path"])
            else:
                # Fallback: reconstruct from annotation records
                canonical_phones = [
                    r["canonical"] for r in mdd_records
                    if r["status"] != "I" and r.get("canonical")
                ]

        # Remove silences
        actual_phones    = [p for p in actual_phones    if p != "sil"]
        canonical_phones = [p for p in canonical_phones if p != "sil"]

        if len(actual_phones) == 0:
            actual_phones = ["sil"]
        if len(canonical_phones) == 0:
            canonical_phones = ["sil"]

        # ── Build 35 CTC label sequences from actual pronounced phonemes ───
        feature_seqs = phoneme_sequence_to_feature_sequences(actual_phones)
        ctc_labels = feature_sequences_to_ctc_labels(feature_seqs)

        return {
            "waveform": waveform,               # (T,)
            "ctc_labels": ctc_labels,           # list[35][U] — from actual_phones
            "actual_phones": actual_phones,     # what speaker said → CTC target
            "canonical_phones": canonical_phones, # what should be said → MDD ref
            "mdd_records": mdd_records,
            "speaker": item["speaker"],
            "utt_id": item["utt_id"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of dataset items into a batch.
    Pads waveforms; keeps ctc_labels as nested lists (handled by loss).
    """
    # Sort by length descending (helps CTC)
    batch = sorted(batch, key=lambda x: x["waveform"].shape[0], reverse=True)

    waveforms = [item["waveform"] for item in batch]
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)

    # Pad waveforms
    max_len = lengths.max().item()
    padded = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w

    return {
        "input_values": padded,             # (B, T)
        "input_lengths": lengths,           # (B,)
        "ctc_labels": [item["ctc_labels"] for item in batch],
        "actual_phones": [item["actual_phones"] for item in batch],
        "canonical_phones": [item["canonical_phones"] for item in batch],
        "mdd_records": [item["mdd_records"] for item in batch],
        "speaker": [item["speaker"] for item in batch],
        "utt_id": [item["utt_id"] for item in batch],
    }


def get_train_test_datasets(
    l2arctic_root: str,
    split: str = "scripted",
    max_duration: float = 15.0,
) -> tuple[L2ArcticDataset, L2ArcticDataset]:
    """
    Build train/test datasets following the paper's split.
    Test = SCRIPTED_TEST_SPEAKERS (one per L1); Train = all others.
    """
    # Discover all speakers
    root = Path(l2arctic_root)
    all_speakers = sorted([
        d.name for d in root.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and d.name != "suitcase_corpus"
    ])

    if len(all_speakers) == 0:
        raise ValueError(f"No speaker directories found in {l2arctic_root}. "
                         "Please download L2-ARCTIC first.")

    test_speakers = [s for s in all_speakers if s in SCRIPTED_TEST_SPEAKERS]
    train_speakers = [s for s in all_speakers if s not in SCRIPTED_TEST_SPEAKERS]

    print(f"Train speakers ({len(train_speakers)}): {train_speakers}")
    print(f"Test  speakers ({len(test_speakers)}):  {test_speakers}")

    train_ds = L2ArcticDataset(l2arctic_root, train_speakers, split, max_duration)
    test_ds  = L2ArcticDataset(l2arctic_root, test_speakers,  split, max_duration)
    return train_ds, test_ds

# ─────────────────────────────────────────────────────────────────────────────
# L2-Suitcase Dataset
# Flat structure: l2arctic/suitcase_corpus/wav/<spk>.wav
#                 l2arctic/suitcase_corpus/annotation/<spk>.TextGrid
# One long recording per speaker — used for TEST ONLY
# ASI did not record suitcase, so suitcase test speakers differ from scripted
# ─────────────────────────────────────────────────────────────────────────────


def _parse_textgrid_with_timestamps(textgrid_path: str):
    """
    Parse suitcase TextGrid and return per-interval records with timestamps.

    Returns list of dicts:
        xmin, xmax, actual_phone, canonical_phone, status, pronounced
    Silence and empty intervals are included so we can find chunk boundaries.
    """
    with open(textgrid_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    tier_blocks = re.split(r'item\s*\[\d+\]', content)
    phones_block = None
    for block in tier_blocks:
        if re.search(r'name\s*=\s*"phones?"', block, re.IGNORECASE):
            phones_block = block
            break

    if phones_block is None:
        return []

    intervals = re.findall(
        r'intervals\s*\[\d+\].*?xmin\s*=\s*([\d.]+).*?xmax\s*=\s*([\d.]+).*?text\s*=\s*"([^"]*)"',
        phones_block, re.DOTALL
    )

    records = []
    for xmin, xmax, text in intervals:
        xmin, xmax = float(xmin), float(xmax)
        text = text.strip()

        # Silence/empty — mark as boundary
        if text in ("", "sil", "sp", "spn", "<unk>"):
            records.append({
                "xmin": xmin, "xmax": xmax,
                "is_silence": True,
                "actual_phone": "sil", "canonical_phone": "sil",
                "status": "C", "pronounced": None,
            })
            continue

        parts = [p.strip() for p in text.split(",")]

        if len(parts) == 1:
            ph = normalize_phoneme(parts[0])
            records.append({
                "xmin": xmin, "xmax": xmax,
                "is_silence": ph == "sil",
                "actual_phone": ph, "canonical_phone": ph,
                "status": "C", "pronounced": None,
            })

        elif len(parts) == 3:
            canonical_raw, pronounced_raw, error_type = parts
            error_type = error_type.strip().lower()
            canonical = normalize_phoneme(canonical_raw)

            if error_type == "s":
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                pronounced = None if pronounced_clean.lower() == "err" else normalize_phoneme(pronounced_clean)
                if pronounced == "sil": pronounced = None
                actual = pronounced if pronounced else canonical
                records.append({
                    "xmin": xmin, "xmax": xmax, "is_silence": False,
                    "actual_phone": actual, "canonical_phone": canonical,
                    "status": "S", "pronounced": pronounced,
                })
            elif error_type == "d":
                records.append({
                    "xmin": xmin, "xmax": xmax, "is_silence": False,
                    "actual_phone": None, "canonical_phone": canonical,
                    "status": "D", "pronounced": None,
                })
            elif error_type == "a":
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                pronounced = normalize_phoneme(pronounced_clean)
                if pronounced == "sil": pronounced = None
                records.append({
                    "xmin": xmin, "xmax": xmax, "is_silence": False,
                    "actual_phone": pronounced, "canonical_phone": None,
                    "status": "I", "pronounced": pronounced,
                })

    return records


def _chunk_suitcase_records(
    records: list[dict],
    max_chunk_duration: float = 10.0,
) -> list[list[dict]]:
    """
    Split per-interval records into chunks of max_chunk_duration seconds.
    Chunk boundaries are placed at silence intervals when possible.
    """
    if not records:
        return []

    chunks = []
    current_chunk = []
    chunk_start = records[0]["xmin"]

    for rec in records:
        current_duration = rec["xmax"] - chunk_start

        # If we hit a silence and chunk is long enough → start new chunk
        if rec["is_silence"] and current_duration >= max_chunk_duration * 0.5:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = []
            chunk_start = rec["xmax"]
            continue

        # If chunk exceeds max duration → force split
        if current_duration >= max_chunk_duration:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = [rec]
            chunk_start = rec["xmin"]
        else:
            current_chunk.append(rec)

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


class SuitcaseDataset(Dataset):
    """
    Dataset for L2-Arctic suitcase (spontaneous speech) corpus.
    One wav + one TextGrid per speaker, chunked into ~10s segments
    using TextGrid timestamps. Used for testing only.
    ASI excluded (did not record suitcase).
    """

    def __init__(
        self,
        l2arctic_root: str,
        speakers: list[str],
        max_chunk_duration: float = 10.0,
        sample_rate: int = 16000,
    ):
        self.root        = Path(l2arctic_root) / "suitcase_corpus"
        self.speakers    = speakers
        self.max_chunk   = max_chunk_duration
        self.sample_rate = sample_rate

        self.samples = self._collect_samples()
        print(f"[SuitcaseDataset] speakers={len(speakers)}, "
              f"chunks={len(self.samples)}")

    def _collect_samples(self) -> list[dict]:
        """
        For each speaker, load the long recording and TextGrid,
        then split into chunks. Each chunk becomes one dataset item.
        Canonical phones come from the transcript .txt file via CMUdict.
        """
        samples = []
        wav_dir = self.root / "wav"
        ann_dir = self.root / "annotation"
        trs_dir = self.root / "transcript"

        for spk in self.speakers:
            wav_file = wav_dir / f"{spk.lower()}.wav"
            tg_file  = ann_dir / f"{spk.lower()}.TextGrid"
            trs_file = trs_dir / f"{spk.lower()}.txt"

            if not wav_file.exists():
                print(f"  [warn] suitcase wav not found: {wav_file}")
                continue
            if not tg_file.exists():
                print(f"  [warn] suitcase TextGrid not found: {tg_file}")
                continue

            # Load full canonical phone sequence from transcript (CMUdict)
            # This is the authoritative reference, independent of annotation.
            if trs_file.exists():
                with open(str(trs_file), "r", encoding="utf-8") as f:
                    transcript_text = f.read().strip()
                all_canonical_phones = text_to_phones(transcript_text)
            else:
                print(f"  [warn] suitcase transcript not found: {trs_file}")
                all_canonical_phones = []

            # Load full audio once to get sample rate and total length
            _, native_sr = torchaudio.load(str(wav_file), num_frames=1)

            # Parse TextGrid with timestamps
            records = _parse_textgrid_with_timestamps(str(tg_file))
            if not records:
                continue

            # Split into chunks — chunking logic unchanged
            chunks = _chunk_suitcase_records(records, self.max_chunk)

            # Map transcript canonical phones to chunks by proportion of
            # total non-silence phones. This is approximate but principled
            # since the transcript is one continuous narrative.
            total_non_sil = sum(
                1 for r in records
                if not r["is_silence"] and r.get("actual_phone")
                and r["actual_phone"] != "sil"
            )

            phone_cursor = 0  # tracks position in all_canonical_phones

            for chunk_idx, chunk_records in enumerate(chunks):
                # Skip empty or all-silence chunks
                non_sil = [r for r in chunk_records if not r["is_silence"]]
                if not non_sil:
                    continue

                # Audio slice boundaries in samples
                start_sec = chunk_records[0]["xmin"]
                end_sec   = chunk_records[-1]["xmax"]
                start_sample = int(start_sec * native_sr)
                end_sample   = int(end_sec   * native_sr)

                # Build phone sequences for this chunk from annotation
                actual_phones = []
                mdd_records   = []

                chunk_non_sil_count = 0
                for r in chunk_records:
                    if r["is_silence"]:
                        continue
                    if r["actual_phone"] and r["actual_phone"] != "sil":
                        actual_phones.append(r["actual_phone"])
                        chunk_non_sil_count += 1
                    mdd_records.append({
                        "canonical":  r["canonical_phone"],
                        "status":     r["status"],
                        "pronounced": r["pronounced"],
                    })

                if not actual_phones:
                    continue

                # Assign canonical phones from transcript proportionally
                # Take the next N canonical phones where N = chunk non-silence count
                if all_canonical_phones:
                    end_cursor = min(
                        phone_cursor + chunk_non_sil_count,
                        len(all_canonical_phones)
                    )
                    canonical_phones = all_canonical_phones[phone_cursor:end_cursor]
                    phone_cursor = end_cursor
                else:
                    # Fallback: reconstruct from annotation (less reliable)
                    canonical_phones = [
                        r["canonical_phone"] for r in chunk_records
                        if not r["is_silence"]
                        and r["canonical_phone"]
                        and r["canonical_phone"] != "sil"
                    ]

                samples.append({
                    "wav_path":        str(wav_file),
                    "native_sr":       native_sr,
                    "start_sample":    start_sample,
                    "end_sample":      end_sample,
                    "actual_phones":   actual_phones,
                    "canonical_phones": canonical_phones,
                    "mdd_records":     mdd_records,
                    "speaker":         spk,
                    "utt_id":          f"{spk.lower()}_{chunk_idx:03d}",
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        # Load only the chunk slice from the full audio
        waveform, sr = torchaudio.load(
            item["wav_path"],
            frame_offset=item["start_sample"],
            num_frames=item["end_sample"] - item["start_sample"],
        )
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)

        actual_phones    = item["actual_phones"]
        canonical_phones = item["canonical_phones"]
        mdd_records      = item["mdd_records"]

        feature_seqs = phoneme_sequence_to_feature_sequences(actual_phones)
        ctc_labels   = feature_sequences_to_ctc_labels(feature_seqs)

        return {
            "waveform":         waveform,
            "ctc_labels":       ctc_labels,
            "actual_phones":    actual_phones,
            "canonical_phones": canonical_phones,
            "mdd_records":      mdd_records,
            "speaker":          item["speaker"],
            "utt_id":           item["utt_id"],
        }



def get_suitcase_train_test_datasets(
    l2arctic_root: str,
    max_chunk_duration: float = 10.0,
) -> tuple["SuitcaseDataset", "SuitcaseDataset"]:
    """
    Build suitcase train/test datasets.
    Suitcase has 22 speakers (ASI and SKA excluded — did not record suitcase).
    Test = SUITCASE_TEST_SPEAKERS (RRBI replaces ASI because ASI has no suitcase recording).
    Train = remaining 16 suitcase speakers.
    """
    # Only use speakers that actually have suitcase recordings
    wav_dir = Path(l2arctic_root) / "suitcase_corpus" / "wav"
    suitcase_speakers = sorted([
        f.stem.upper() for f in wav_dir.glob("*.wav")
    ])
    print(f"Suitcase speakers found: {len(suitcase_speakers)}: {suitcase_speakers}")

    suitcase_test_speakers  = [s for s in suitcase_speakers if s in SUITCASE_TEST_SPEAKERS]
    suitcase_train_speakers = [s for s in suitcase_speakers if s not in SUITCASE_TEST_SPEAKERS]

    print(f"Suitcase train speakers ({len(suitcase_train_speakers)}): {suitcase_train_speakers}")
    print(f"Suitcase test  speakers ({len(suitcase_test_speakers)}):  {suitcase_test_speakers}")

    train_ds = SuitcaseDataset(l2arctic_root, suitcase_train_speakers, max_chunk_duration)
    test_ds  = SuitcaseDataset(l2arctic_root, suitcase_test_speakers,  max_chunk_duration)
    return train_ds, test_ds

def get_suitcase_test_dataset(
    l2arctic_root: str,
    max_chunk_duration: float = 10.0,
) -> "SuitcaseDataset":
    """Return suitcase test dataset — SUITCASE_TEST_SPEAKERS."""
    wav_dir = Path(l2arctic_root) / "suitcase_corpus" / "wav"
    suitcase_speakers = sorted([f.stem.upper() for f in wav_dir.glob("*.wav")])
    test_speakers = [s for s in suitcase_speakers if s in SUITCASE_TEST_SPEAKERS]
    print(f"Suitcase test speakers ({len(test_speakers)}): {test_speakers}")
    return SuitcaseDataset(l2arctic_root, test_speakers, max_chunk_duration)
