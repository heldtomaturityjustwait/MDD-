import os
import json
import torch
import torchaudio
import argparse
from tqdm import tqdm

# ----------------------------
# SIMPLE LEVENSHTEIN
# ----------------------------
def levenshtein(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[-1][-1]


# ----------------------------
# CTC COLLAPSE
# ----------------------------
def ctc_collapse(seq, blank):
    result = []
    prev = None
    for s in seq:
        if s != blank and s != prev:
            result.append(s)
        prev = s
    return result


# ----------------------------
# DECODE SCTC-SB
# ----------------------------
def decode_sctc_sb(logits):
    """
    logits: (T, 71)
    returns: list of 35 feature sequences
    """
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1)  # (T,)

    blank = 70
    feature_sequences = []

    for f in range(35):
        pos_idx = f
        neg_idx = f + 35

        seq = []
        for t in pred:
            if t == pos_idx:
                seq.append(1)
            elif t == neg_idx:
                seq.append(0)
            else:
                seq.append(-1)  # blank/irrelevant

        # remove -1
        seq = [s for s in seq if s != -1]

        seq = ctc_collapse(seq, blank=None)
        feature_sequences.append(seq)

    return feature_sequences


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model(path, device):
    model = torch.load(path, map_location=device)
    model.eval()
    return model


# ----------------------------
# LOAD AUDIO
# ----------------------------
def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


# ----------------------------
# MAIN
# ----------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.sctcSB_model, device)

    total_FAR = 0
    total_FRR = 0
    total_DER = 0
    total_count = 0

    results = []

    speakers = os.listdir(args.l2arctic_dir)

    for spk in tqdm(speakers):
        wav_dir = os.path.join(args.l2arctic_dir, spk, "wav")

        for file in os.listdir(wav_dir):
            if not file.endswith(".wav"):
                continue

            wav_path = os.path.join(wav_dir, file)

            audio = load_audio(wav_path).to(device)
            audio = audio.unsqueeze(0)

            with torch.no_grad():
                logits = model(audio)[0]  # (T,71)

            pred_feats = decode_sctc_sb(logits)

            # ⚠️ placeholder canonical (you MUST replace)
            # load canonical feature sequence here
            canonical_feats = pred_feats  # TEMP (for debug)

            # ----------------------------
            # MDD METRICS
            # ----------------------------
            FAR = 0
            FRR = 0
            DER = 0

            for f in range(35):
                p = pred_feats[f]
                c = canonical_feats[f]

                dist = levenshtein(p, c)

                DER += dist

                for i in range(min(len(p), len(c))):
                    if c[i] == 1 and p[i] == 0:
                        FRR += 1
                    elif c[i] == 0 and p[i] == 1:
                        FAR += 1

            total_FAR += FAR
            total_FRR += FRR
            total_DER += DER
            total_count += 1

            results.append({
                "file": file,
                "FAR": FAR,
                "FRR": FRR,
                "DER": DER
            })

    final = {
        "avg_FAR": total_FAR / total_count,
        "avg_FRR": total_FRR / total_count,
        "avg_DER": total_DER / total_count,
        "num_samples": total_count
    }

    with open(args.output_json, "w") as f:
        json.dump({"summary": final, "details": results}, f, indent=2)

    print(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l2arctic_dir", type=str, required=True)
    parser.add_argument("--sctcSB_model", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)

    args = parser.parse_args()
    main(args)