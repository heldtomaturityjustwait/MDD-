import os
import json
import torch
import torchaudio
import argparse
from tqdm import tqdm
from praatio import textgrid

# ----------------------------
# PHONEME → FEATURE MAP (SIMPLE)
# ----------------------------
# You can refine later
PHONEME_FEATURE_MAP = {
    "AA": [1]*35,
    "AE": [1]*35,
    "AH": [1]*35,
    "B":  [0]*35,
    "P":  [0]*35,
    # TODO: extend properly
}

# ----------------------------
# LOAD AUDIO
# ----------------------------
def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)

# ----------------------------
# LOAD TEXTGRID → PHONEMES
# ----------------------------
def load_canonical_phonemes(tg_path):
    tg = textgrid.openTextgrid(tg_path, includeEmptyIntervals=False)
    
    tier_name = tg.tierNameList[0]  # usually phones
    tier = tg.tierDict[tier_name]

    phonemes = []
    for entry in tier.entryList:
        label = entry.label.strip()
        if label and label != "sil":
            phonemes.append(label.upper())
    
    return phonemes

# ----------------------------
# PHONEMES → FEATURES
# ----------------------------
def phonemes_to_features(phonemes):
    feats = []
    for p in phonemes:
        if p in PHONEME_FEATURE_MAP:
            feats.append(PHONEME_FEATURE_MAP[p])
        else:
            feats.append([0]*35)
    return feats

# ----------------------------
# CTC COLLAPSE
# ----------------------------
def ctc_collapse(seq):
    out = []
    prev = None
    for s in seq:
        if s != prev:
            out.append(s)
        prev = s
    return out

# ----------------------------
# DECODE SCTC-SB
# ----------------------------
def decode_sctc(logits):
    pred = torch.argmax(logits, dim=-1)
    blank = 70

    feature_seqs = []

    for f in range(35):
        pos = f
        neg = f + 35

        seq = []
        for t in pred:
            if t == pos:
                seq.append(1)
            elif t == neg:
                seq.append(0)

        seq = ctc_collapse(seq)
        feature_seqs.append(seq)

    return feature_seqs

# ----------------------------
# LEVENSHTEIN
# ----------------------------
def levenshtein(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1]+cost
            )
    return dp[-1][-1]

# ----------------------------
# MAIN
# ----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(args.sctcSB_model, map_location=device)
    model.eval()

    speaker_dir = os.path.join(args.l2arctic_dir, args.test_speaker)

    wav_dir = os.path.join(speaker_dir, "wav")
    tg_dir  = os.path.join(speaker_dir, "textgrid")

    total_FAR = 0
    total_FRR = 0
    total_DER = 0
    count = 0

    results = []

    files = os.listdir(wav_dir)

    for f in tqdm(files):
        if not f.endswith(".wav"):
            continue

        wav_path = os.path.join(wav_dir, f)
        tg_path  = os.path.join(tg_dir, f.replace(".wav", ".TextGrid"))

        if not os.path.exists(tg_path):
            continue

        # ----------------------------
        # LOAD DATA
        # ----------------------------
        audio = load_audio(wav_path).to(device).unsqueeze(0)

        phonemes = load_canonical_phonemes(tg_path)
        canonical_feats = phonemes_to_features(phonemes)

        # ----------------------------
        # MODEL
        # ----------------------------
        with torch.no_grad():
            logits = model(audio)[0]

        pred_feats = decode_sctc(logits)

        # ----------------------------
        # METRICS
        # ----------------------------
        FAR = 0
        FRR = 0
        DER = 0

        for i in range(35):
            p = pred_feats[i]
            c = [f[i] for f in canonical_feats]

            dist = levenshtein(p, c)
            DER += dist

            for j in range(min(len(p), len(c))):
                if c[j] == 1 and p[j] == 0:
                    FRR += 1
                elif c[j] == 0 and p[j] == 1:
                    FAR += 1

        total_FAR += FAR
        total_FRR += FRR
        total_DER += DER
        count += 1

        results.append({
            "file": f,
            "FAR": FAR,
            "FRR": FRR,
            "DER": DER
        })

    summary = {
        "speaker": args.test_speaker,
        "avg_FAR": total_FAR / max(count,1),
        "avg_FRR": total_FRR / max(count,1),
        "avg_DER": total_DER / max(count,1)
    }

    with open(args.output_json, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2arctic_dir", required=True)
    parser.add_argument("--sctcSB_model", required=True)
    parser.add_argument("--test_speaker", required=True)
    parser.add_argument("--output_json", required=True)

    args = parser.parse_args()
    main(args)