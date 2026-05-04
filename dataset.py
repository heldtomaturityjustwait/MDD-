"""
dataset.py
==========
PyTorch Datasets for phonological feature recognition training.

Data sources:
  1. TIMIT        — fully annotated, used for training only
  2. L2-ARCTIC    — human-annotated utterances (annotation/ TextGrids only)
                    ~150 files per speaker; unannotated utterances are SKIPPED
  3. L2-Suitcase  — spontaneous speech, fully annotated TextGrid per speaker

Training target: actual spoken phonemes → 35 phonological feature sequences
(via phonological_features.py).  Canonical phones and MDD records are NOT
used — this module only cares about what was actually said.

TIMIT directory structure:
    timit/
      TRAIN/  TEST/
        DR{1-8}/
          <SPEAKER_ID>/
            <UTT_ID>.WAV   (or .wav — both handled)
            <UTT_ID>.PHN   (sample-start  sample-end  phoneme, one per line)

L2-ARCTIC directory structure:
    l2arctic/
      <SPEAKER_ID>/
        wav/          ← utterance .wav files (16kHz)
        annotation/   ← human-annotated .TextGrid files (~150 per speaker)
                        Contains C/S/D/I error labels.  Only these files are
                        used — unannotated utterances are skipped entirely.
      suitcase_corpus/
        wav/          ← one .wav per speaker (long, spontaneous)
        annotation/   ← one .TextGrid per speaker (fully annotated C/S/D/I)
"""

import re
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset

from phonological_features import (
    CMU_39_PHONEMES,
    phoneme_sequence_to_feature_sequences,
    feature_sequences_to_ctc_labels,
)

# ─────────────────────────────────────────────────────────────────────────────
# Speaker splits (Ye et al. 2022 / Shahin et al. 2025)
# ─────────────────────────────────────────────────────────────────────────────
SCRIPTED_TEST_SPEAKERS  = {"RRBI", "YBAA", "HJK", "BWC", "EBVS", "YDCK"}
SUITCASE_TEST_SPEAKERS  = {"RRBI", "YBAA", "HJK", "BWC", "EBVS", "YDCK"}

# TIMIT phone labels that need remapping to the CMU 39-phoneme set
# (TIMIT 61-phone → 39-phone reduction, standard mapping)
_TIMIT_REMAP = {
    "ax":  "ah", "ax-h": "ah",
    "ix":  "ih",
    "nx":  "n",
    "em":  "m",
    "en":  "n",  "eng": "ng",
    "el":  "l",
    "zh":  "zh",
    "pcl": "sil", "tcl": "sil", "kcl": "sil",
    "bcl": "sil", "dcl": "sil", "gcl": "sil",
    "epi": "sil", "pau": "sil", "h#": "sil",
    "q":   "sil",  # glottal stop → silence
    "hv":  "hh",
    "ax-r": "er",
    "ux":  "uw",
    "ow":  "ow",
}


# ─────────────────────────────────────────────────────────────────────────────
# Phoneme normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_phoneme(ph: str) -> str:
    """
    Strip stress markers, lowercase, apply known remaps.
    Returns 'sil' for anything outside the CMU 39-phoneme set.
    """
    ph = ph.lower().strip()
    ph = re.sub(r"[0-9]", "", ph)   # strip ARPAbet stress digits
    ph = _TIMIT_REMAP.get(ph, ph)   # TIMIT-specific remaps
    if ph not in CMU_39_PHONEMES:
        return "sil"
    return ph


# ─────────────────────────────────────────────────────────────────────────────
# TIMIT .PHN parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_phn_file(phn_path: str) -> list[str]:
    """
    Parse a TIMIT .PHN file and return the phoneme sequence.

    File format (one interval per line):
        <start_sample>  <end_sample>  <phoneme_label>

    Silence tokens (h#, pau, epi, pcl, …) are dropped; the list contains
    only the speech phonemes, in order.
    """
    phones = []
    with open(phn_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            ph = normalize_phoneme(parts[2])
            if ph != "sil":
                phones.append(ph)
    return phones


# ─────────────────────────────────────────────────────────────────────────────
# L2-ARCTIC annotation TextGrid parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_annotation_textgrid(textgrid_path: str) -> list[str]:
    """
    Extract the actual spoken phoneme sequence from an L2-ARCTIC
    human-annotation TextGrid (annotation/ folder).

    TextGrid phones tier label formats:
        Correct:      "AH1"          → spoken = ah
        Substitution: "DH,D,s"       → spoken = d   (what was said)
        Deletion:     "TH,sil,d"     → speaker said nothing → skip
        Addition:     "sil,AH,a"     → spoken = ah  (extra phoneme inserted)
        Hard error:   "CPL,err,s"    → skip (uninterpretable)

    Returns only the phonemes that were actually produced by the speaker,
    in order, silences removed.
    """
    with open(textgrid_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Find the phones tier
    tier_blocks = re.split(r'item\s*\[\d+\]', content)
    phones_block = None
    for block in tier_blocks:
        if re.search(r'name\s*=\s*"phones?"', block, re.IGNORECASE):
            phones_block = block
            break
    if phones_block is None:
        return []

    intervals = re.findall(
        r'intervals\s*\[\d+\].*?xmin\s*=\s*[\d.]+.*?xmax\s*=\s*[\d.]+'
        r'.*?text\s*=\s*"([^"]*)"',
        phones_block, re.DOTALL,
    )

    actual_phones = []
    for text in intervals:
        text = text.strip()
        if text in ("", "sil", "sp", "spn", "<unk>"):
            continue

        parts = [p.strip() for p in text.split(",")]

        if len(parts) == 1:
            # Correct pronunciation
            ph = normalize_phoneme(parts[0])
            if ph != "sil":
                actual_phones.append(ph)

        elif len(parts) == 3:
            canonical_raw, pronounced_raw, error_type = parts
            error_type = error_type.strip().lower()

            if error_type == "s":
                # Substitution — use what was actually pronounced
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                if pronounced_clean.lower() == "err":
                    # Uninterpretable error: fall back to canonical
                    ph = normalize_phoneme(canonical_raw)
                else:
                    ph = normalize_phoneme(pronounced_clean)
                if ph != "sil":
                    actual_phones.append(ph)

            elif error_type == "d":
                # Deletion — nothing was said, skip
                pass

            elif error_type == "a":
                # Addition — extra phoneme inserted by the speaker
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                ph = normalize_phoneme(pronounced_clean)
                if ph != "sil":
                    actual_phones.append(ph)

    return actual_phones


# ─────────────────────────────────────────────────────────────────────────────
# Suitcase TextGrid parser (timestamp-aware, for chunking)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_suitcase_textgrid(textgrid_path: str) -> list[dict]:
    """
    Parse a suitcase annotation TextGrid, returning per-interval records
    with timestamps so we can split the long recording into chunks.

    Each record:  {xmin, xmax, is_silence, actual_phone}
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
        r'intervals\s*\[\d+\].*?xmin\s*=\s*([\d.]+).*?xmax\s*=\s*([\d.]+)'
        r'.*?text\s*=\s*"([^"]*)"',
        phones_block, re.DOTALL,
    )

    records = []
    for xmin, xmax, text in intervals:
        xmin, xmax = float(xmin), float(xmax)
        text = text.strip()

        if text in ("", "sil", "sp", "spn", "<unk>"):
            records.append({"xmin": xmin, "xmax": xmax,
                            "is_silence": True, "actual_phone": None})
            continue

        parts = [p.strip() for p in text.split(",")]
        actual = None

        if len(parts) == 1:
            ph = normalize_phoneme(parts[0])
            actual = ph if ph != "sil" else None

        elif len(parts) == 3:
            canonical_raw, pronounced_raw, error_type = parts
            error_type = error_type.strip().lower()
            if error_type == "s":
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                ph = (normalize_phoneme(canonical_raw)
                      if pronounced_clean.lower() == "err"
                      else normalize_phoneme(pronounced_clean))
                actual = ph if ph != "sil" else None
            elif error_type == "d":
                actual = None   # deletion → nothing said
            elif error_type == "a":
                pronounced_clean = pronounced_raw.replace("*", "").strip()
                ph = normalize_phoneme(pronounced_clean)
                actual = ph if ph != "sil" else None

        is_silence = actual is None
        records.append({"xmin": xmin, "xmax": xmax,
                        "is_silence": is_silence, "actual_phone": actual})
    return records


def _chunk_records(
    records: list[dict],
    max_chunk_duration: float = 10.0,
) -> list[list[dict]]:
    """
    Split timestamp records into chunks ≤ max_chunk_duration seconds.
    Prefers silence boundaries; force-splits when the limit is exceeded.
    """
    if not records:
        return []
    chunks, current, chunk_start = [], [], records[0]["xmin"]
    for rec in records:
        duration = rec["xmax"] - chunk_start
        if rec["is_silence"] and duration >= max_chunk_duration * 0.5:
            if current:
                chunks.append(current)
            current, chunk_start = [], rec["xmax"]
            continue
        if duration >= max_chunk_duration:
            if current:
                chunks.append(current)
            current, chunk_start = [rec], rec["xmin"]
        else:
            current.append(rec)
    if current:
        chunks.append(current)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _phones_to_ctc_labels(phones: list[str]) -> list[list[int]]:
    """phones → 35 CTC label sequences."""
    return feature_sequences_to_ctc_labels(
        phoneme_sequence_to_feature_sequences(phones)
    )


# ─────────────────────────────────────────────────────────────────────────────
# TIMIT Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TIMITDataset(Dataset):
    """
    TIMIT dataset for phonological feature pre-training.

    Reads all .WAV / .wav + .PHN files under timit_root/TRAIN or TEST.
    Each item returns:
        waveform      : (T,)  float32 at 16kHz
        ctc_labels    : list[35][U]  CTC label sequences
        actual_phones : list[str]    recognised phoneme sequence

    Args:
        timit_root  : path to TIMIT root (contains TRAIN/ and TEST/)
        split       : "TRAIN" or "TEST"
        max_duration: max audio length in seconds (longer clips are truncated)
        sample_rate : target sample rate (TIMIT native = 16kHz)
    """

    def __init__(
        self,
        timit_root: str,
        split: str = "TRAIN",
        max_duration: float = 15.0,
        sample_rate: int = 16000,
    ):
        self.root        = Path(timit_root) / split.upper()
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self.samples     = self._collect_samples()
        print(f"[TIMITDataset] split={split.upper()}, "
              f"utterances={len(self.samples)}")

    def _collect_samples(self) -> list[dict]:
        samples = []
        # rglob is case-sensitive on Linux — try both .PHN and .phn
        phn_files = sorted(self.root.rglob("*.PHN"))
        if not phn_files:
            phn_files = sorted(self.root.rglob("*.phn"))

        for phn_file in phn_files:
            # Skip SA sentences — same 2 prompts read by all 630 speakers,
            # would leak identical text across train/test splits
            if phn_file.stem.lower().startswith("sa"):
                continue

            # Match wav: handles SX217.WAV, SX127.WAV.wav, sx217.wav, etc.
            stem = phn_file.stem   # e.g. "SX37" or "SI1027"
            parent = phn_file.parent
            wav_file = None
            for candidate in parent.iterdir():
                cname = candidate.name.lower()
                cstem = cname.split(".")[0]   # strip all extensions
                if cstem == stem.lower() and cname.endswith((".wav",)):
                    wav_file = candidate
                    break
            if wav_file is None:
                continue

            phones = parse_phn_file(str(phn_file))
            if not phones:
                continue

            samples.append({
                "wav_path": str(wav_file),
                "phones":   phones,
                "utt_id":   phn_file.stem,
                "speaker":  phn_file.parent.name,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        # TIMIT .WAV files are NIST sphere format — try torchaudio first,
        # fall back to scipy which handles sphere natively
        try:
            waveform, sr = torchaudio.load(item["wav_path"])
        except Exception:
            import scipy.io.wavfile as sciwav
            import numpy as np
            sr, data = sciwav.read(item["wav_path"])
            if data.dtype != np.float32:
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            waveform = torch.from_numpy(data).unsqueeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]

        phones = item["phones"]
        return {
            "waveform":      waveform,
            "ctc_labels":    _phones_to_ctc_labels(phones),
            "actual_phones": phones,
            "speaker":       item["speaker"],
            "utt_id":        item["utt_id"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# L2-ARCTIC Scripted Dataset
# ─────────────────────────────────────────────────────────────────────────────

class L2ArcticDataset(Dataset):
    """
    L2-ARCTIC scripted speech dataset.

    Only utterances with a human annotation TextGrid (annotation/ folder)
    are included.  Unannotated utterances are skipped entirely — we do NOT
    fall back to forced-alignment textgrids.

    Each item returns:
        waveform      : (T,)  float32 at 16kHz
        ctc_labels    : list[35][U]
        actual_phones : list[str]   what the speaker actually said
        speaker       : str
        utt_id        : str
    """

    def __init__(
        self,
        l2arctic_root: str,
        speakers: list[str],
        max_duration: float = 15.0,
        sample_rate: int = 16000,
    ):
        self.root        = Path(l2arctic_root)
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self.samples     = self._collect_samples(speakers)
        print(f"[L2ArcticDataset] speakers={len(speakers)}, "
              f"annotated_utterances={len(self.samples)}")

    def _collect_samples(self, speakers: list[str]) -> list[dict]:
        samples  = []
        n_skip   = 0

        for spk in speakers:
            wav_dir = self.root / spk / "wav"
            ann_dir = self.root / spk / "annotation"

            if not wav_dir.exists():
                print(f"  [warn] wav dir missing: {wav_dir}")
                continue
            if not ann_dir.exists():
                print(f"  [warn] annotation dir missing: {ann_dir}")
                continue

            for ann_file in sorted(ann_dir.glob("*.TextGrid")):
                utt_id   = ann_file.stem
                wav_file = wav_dir / f"{utt_id}.wav"
                if not wav_file.exists():
                    n_skip += 1
                    continue

                phones = parse_annotation_textgrid(str(ann_file))
                if not phones:
                    n_skip += 1
                    continue

                samples.append({
                    "wav_path":      str(wav_file),
                    "phones":        phones,
                    "speaker":       spk,
                    "utt_id":        utt_id,
                })

        print(f"  [L2ArcticDataset] skipped (no wav / empty)={n_skip}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        waveform, sr = torchaudio.load(item["wav_path"])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]

        phones = item["phones"]
        return {
            "waveform":      waveform,
            "ctc_labels":    _phones_to_ctc_labels(phones),
            "actual_phones": phones,
            "speaker":       item["speaker"],
            "utt_id":        item["utt_id"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# L2-ARCTIC Suitcase Dataset
# ─────────────────────────────────────────────────────────────────────────────

class SuitcaseDataset(Dataset):
    """
    L2-ARCTIC suitcase (spontaneous speech) dataset.

    One long WAV + one fully-annotated TextGrid per speaker.
    The recording is split into chunks of ≤ max_chunk_duration seconds
    using silence boundaries from the TextGrid timestamps.

    Each item returns:
        waveform      : (T,)  float32 at 16kHz
        ctc_labels    : list[35][U]
        actual_phones : list[str]
        speaker       : str
        utt_id        : str    e.g. "HJK_007"
    """

    def __init__(
        self,
        l2arctic_root: str,
        speakers: list[str],
        max_chunk_duration: float = 10.0,
        sample_rate: int = 16000,
    ):
        self.sample_rate = sample_rate
        self.samples     = self._collect_samples(
            Path(l2arctic_root) / "suitcase_corpus",
            speakers, max_chunk_duration,
        )
        print(f"[SuitcaseDataset] speakers={len(speakers)}, "
              f"chunks={len(self.samples)}")

    def _collect_samples(
        self,
        root: Path,
        speakers: list[str],
        max_chunk_duration: float,
    ) -> list[dict]:
        samples = []
        wav_dir = root / "wav"
        ann_dir = root / "annotation"

        for spk in speakers:
            wav_file = wav_dir / f"{spk.lower()}.wav"
            tg_file  = ann_dir / f"{spk.lower()}.TextGrid"

            if not wav_file.exists():
                print(f"  [warn] suitcase wav missing: {wav_file}")
                continue
            if not tg_file.exists():
                print(f"  [warn] suitcase TextGrid missing: {tg_file}")
                continue

            _, native_sr = torchaudio.load(str(wav_file), num_frames=1)
            records = _parse_suitcase_textgrid(str(tg_file))
            if not records:
                continue

            for chunk_idx, chunk in enumerate(_chunk_records(records, max_chunk_duration)):
                phones = [r["actual_phone"] for r in chunk
                          if not r["is_silence"] and r["actual_phone"]]
                if not phones:
                    continue

                start_sample = int(chunk[0]["xmin"] * native_sr)
                end_sample   = int(chunk[-1]["xmax"] * native_sr)

                samples.append({
                    "wav_path":    str(wav_file),
                    "native_sr":   native_sr,
                    "start":       start_sample,
                    "end":         end_sample,
                    "phones":      phones,
                    "speaker":     spk,
                    "utt_id":      f"{spk.lower()}_{chunk_idx:03d}",
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        waveform, sr = torchaudio.load(
            item["wav_path"],
            frame_offset=item["start"],
            num_frames=item["end"] - item["start"],
        )
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)

        phones = item["phones"]
        return {
            "waveform":      waveform,
            "ctc_labels":    _phones_to_ctc_labels(phones),
            "actual_phones": phones,
            "speaker":       item["speaker"],
            "utt_id":        item["utt_id"],
        }


# ─────────────────────────────────────────────────────────────────────────────
# collate_fn
# ─────────────────────────────────────────────────────────────────────────────

def make_collate_fn(processor):
    """
    Factory that returns a collate_fn with a Wav2Vec2Processor bound to it.

    The processor handles two things that we previously did manually:
      1. Normalization — zero-mean, unit-variance per waveform
         (required because wav2vec2 was pretrained on normalized audio)
      2. Padding + attention_mask — pads all waveforms in the batch to
         the same length and returns a boolean mask

    Args:
        processor: Wav2Vec2Processor loaded from the same pretrained model

    Returns:
        collate_fn(batch) → dict with keys:
            input_values   : (B, T)       normalized + padded waveforms
            attention_mask : (B, T)       1 = real audio, 0 = padding
            input_lengths  : (B,)         actual sample lengths (for reference)
            ctc_labels     : [B][35][U]   CTC label sequences
            actual_phones  : [B][list[str]]
            speaker        : [B][str]
            utt_id         : [B][str]
    """
    def collate_fn(batch: list[dict]) -> dict:
        # Sort longest-first (standard for CTC batching efficiency)
        batch = sorted(batch, key=lambda x: x["waveform"].shape[0], reverse=True)

        # Raw waveforms as numpy arrays — processor expects list of 1-D arrays
        raw_waveforms = [b["waveform"].numpy() for b in batch]
        input_lengths = torch.tensor(
            [b["waveform"].shape[0] for b in batch], dtype=torch.long
        )

        # Wav2Vec2Processor normalizes each waveform independently
        # (zero-mean, unit-variance) and pads to the longest in the batch.
        # return_tensors="pt" gives us PyTorch tensors directly.
        processed = processor(
            raw_waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,          # pad shorter sequences to longest
        )
        # processed.input_values  : (B, T_max)  normalized + padded
        # processed.attention_mask: (B, T_max)  1=real, 0=padding

        return {
            "input_values":   processed.input_values,    # (B, T)
            "attention_mask": processed.attention_mask,  # (B, T)
            "input_lengths":  input_lengths,             # (B,) raw sample counts
            "ctc_labels":     [b["ctc_labels"]    for b in batch],
            "actual_phones":  [b["actual_phones"] for b in batch],
            "speaker":        [b["speaker"]       for b in batch],
            "utt_id":         [b["utt_id"]        for b in batch],
        }

    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Unified factory function
# ─────────────────────────────────────────────────────────────────────────────

def get_datasets(
    l2arctic_root: str,
    timit_root: str | None = None,
    max_duration: float = 15.0,
    max_chunk_duration: float = 10.0,
) -> tuple["ConcatDataset", "ConcatDataset"]:
    """
    Build a single train dataset and a single test dataset from all sources.

    Train = L2-ARCTIC annotated train speakers
          + L2-ARCTIC suitcase train speakers
          + TIMIT TRAIN (if timit_root is provided)

    Test  = L2-ARCTIC annotated test speakers
          + L2-ARCTIC suitcase test speakers
          (TIMIT TEST is not used — evaluation is always on L2 speakers)

    Args:
        l2arctic_root      : path to L2-ARCTIC root directory
        timit_root         : path to TIMIT root (optional)
        max_duration       : max clip length in seconds for scripted / TIMIT
        max_chunk_duration : max chunk length in seconds for suitcase

    Returns:
        train_ds, test_ds  : ConcatDataset instances ready for DataLoader
    """
    from torch.utils.data import ConcatDataset

    # ── L2-ARCTIC scripted (annotation/ only) ────────────────────────────
    root = Path(l2arctic_root)
    all_speakers = sorted([
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name != "suitcase_corpus"
    ])
    if not all_speakers:
        raise ValueError(f"No speaker directories found in {l2arctic_root}")

    l2_test_spk  = [s for s in all_speakers if s in SCRIPTED_TEST_SPEAKERS]
    l2_train_spk = [s for s in all_speakers if s not in SCRIPTED_TEST_SPEAKERS]

    print(f"L2-ARCTIC scripted train speakers ({len(l2_train_spk)}): {l2_train_spk}")
    print(f"L2-ARCTIC scripted test  speakers ({len(l2_test_spk)}):  {l2_test_spk}")

    l2_train = L2ArcticDataset(l2arctic_root, l2_train_spk, max_duration)
    l2_test  = L2ArcticDataset(l2arctic_root, l2_test_spk,  max_duration)

    # ── L2-ARCTIC suitcase ────────────────────────────────────────────────
    wav_dir = root / "suitcase_corpus" / "wav"
    all_suit_spk = sorted([f.stem.upper() for f in wav_dir.glob("*.wav")])

    suit_test_spk  = [s for s in all_suit_spk if s in SUITCASE_TEST_SPEAKERS]
    suit_train_spk = [s for s in all_suit_spk if s not in SUITCASE_TEST_SPEAKERS]

    print(f"Suitcase train speakers ({len(suit_train_spk)}): {suit_train_spk}")
    print(f"Suitcase test  speakers ({len(suit_test_spk)}):  {suit_test_spk}")

    suit_train = SuitcaseDataset(l2arctic_root, suit_train_spk, max_chunk_duration)
    suit_test  = SuitcaseDataset(l2arctic_root, suit_test_spk,  max_chunk_duration)

    # ── TIMIT (train only) ────────────────────────────────────────────────
    train_parts = [l2_train, suit_train]
    if timit_root and Path(timit_root).exists():
        timit_ds = TIMITDataset(timit_root, split="TRAIN", max_duration=max_duration)
        train_parts.append(timit_ds)
        print(f"TIMIT TRAIN: {len(timit_ds)} utterances added")
    else:
        print("TIMIT not provided — training on L2-ARCTIC only")

    train_ds = ConcatDataset(train_parts)
    test_ds  = ConcatDataset([l2_test, suit_test])

    print(f"\nTotal train: {len(train_ds)} | Total test: {len(test_ds)}")
    return train_ds, test_ds
