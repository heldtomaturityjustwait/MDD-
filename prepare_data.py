"""
prepare_data.py
===============
Download and verify L2-ARCTIC dataset.

L2-ARCTIC is publicly available at:
  https://psi.engr.tamu.edu/l2-arctic-corpus/

Usage:
    python scripts/prepare_data.py --data_dir ./l2arctic

Note: You may need to register/accept a license on the website before
downloading. This script helps verify the structure once downloaded.
"""

import os
import sys
import argparse
from pathlib import Path


L2ARCTIC_SPEAKERS = {
    # Language: [speaker IDs]
    "Hindi":    ["ASI", "RRBI", "SVBI", "TNI"],
    "Arabic":   ["YBAA", "MBMPS", "SKA", "HQTV"],
    "Korean":   ["HJK", "HKK", "CKJP", "JVJP"],
    "Mandarin": ["BWC", "LXC", "NCC", "TXHC"],
    "Spanish":  ["EBVS", "ERMS", "HQTV", "PNV"],
    "Vietnamese": ["YDCK", "YKWK", "THV", "TLV"],
}

SCRIPTED_TEST_SPEAKERS = {"ASI", "YBAA", "HJK", "BWC", "EBVS", "YDCK"}
SUITCASE_TEST_SPEAKERS = {"RRBI", "YBAA", "HJK", "BWC", "EBVS", "YDCK"}

DOWNLOAD_URL = "https://psi.engr.tamu.edu/l2-arctic-corpus/"


def check_dataset_structure(data_dir: str) -> None:
    root = Path(data_dir)
    if not root.exists():
        print(f"ERROR: Directory '{data_dir}' does not exist.")
        print(f"Please download L2-ARCTIC from:\n  {DOWNLOAD_URL}")
        sys.exit(1)

    print(f"\nChecking L2-ARCTIC structure in: {data_dir}\n")

    all_speakers = [d.name for d in root.iterdir() if d.is_dir()]
    print(f"Found {len(all_speakers)} speaker directories: {sorted(all_speakers)}\n")

    expected = set(s for spks in L2ARCTIC_SPEAKERS.values() for s in spks)
    missing = expected - set(all_speakers)
    if missing:
        print(f"WARNING: Missing speakers: {sorted(missing)}")
    else:
        print("All expected speakers found!")

    # Check structure for each speaker
    for spk in sorted(all_speakers):
        spk_dir = root / spk
        wav_dir = spk_dir / "wav"
        ann_dir = spk_dir / "annotation"

        wav_count = len(list(wav_dir.glob("*.wav"))) if wav_dir.exists() else 0
        ann_count = len(list(ann_dir.glob("*.TextGrid"))) if ann_dir.exists() else 0

        role = "TEST " if spk in SCRIPTED_TEST_SPEAKERS else "TRAIN"
        print(f"  [{role}] {spk:<8} wav={wav_count:4d}  annotations={ann_count:4d}  "
              f"{'✓' if wav_count > 0 else '✗'}")

    print("\nStructure check complete.")
    print("\nExpected directory layout:")
    print("  l2arctic/")
    print("    <SPEAKER>/")
    print("      wav/          ← .wav audio files (16kHz)")
    print("      annotation/   ← .TextGrid phoneme annotations")
    print("      transcript/   ← .txt word-level transcripts")
    print("      suitcase_corpus/  ← spontaneous speech subset")
    print("        wav/")
    print("        annotation/")


def print_download_instructions():
    print("\n" + "="*60)
    print("HOW TO DOWNLOAD L2-ARCTIC")
    print("="*60)
    print(f"""
1. Visit: {DOWNLOAD_URL}

2. Fill in the registration form and accept the license.

3. You will receive a download link. The dataset comes as a
   ZIP archive (~10GB total for all 24 speakers).

4. Extract to your target directory, e.g.:
   unzip l2arctic_release_v5.0.zip -d ./l2arctic

5. Verify structure:
   python scripts/prepare_data.py --data_dir ./l2arctic

ON GOOGLE COLAB, the Colab notebook handles this automatically.
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./l2arctic",
                        help="Path to L2-ARCTIC root directory")
    parser.add_argument("--info", action="store_true",
                        help="Print download instructions")
    args = parser.parse_args()

    if args.info:
        print_download_instructions()
        return

    check_dataset_structure(args.data_dir)


if __name__ == "__main__":
    main()
