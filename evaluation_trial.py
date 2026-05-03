import numpy as np

def debug_one_sample(canonical, human, logits):
    from mdd_evaluation import (
        count_phonological_mdd,
        _decode_sctcSB_logits_to_feature_sequences,
        _align_binary_sequence_to_canonical,
        _phoneme_to_binary_array,
    )

    print("\n================ DEBUG MDD =================")
    print("Canonical:", canonical)
    print("Human:    ", human)

    # Step 1: decode logits
    pred_feature_seqs = _decode_sctcSB_logits_to_feature_sequences(logits)

    print("\n--- Feature 0 example ---")
    print("Pred seq (raw):", pred_feature_seqs[0])

    # Step 2: canonical feature
    canon_feats = np.stack([_phoneme_to_binary_array(p) for p in canonical])
    ref_seq = canon_feats[:, 0].tolist()

    print("Canonical feature seq:", ref_seq)

    # Step 3: alignment
    aligned = _align_binary_sequence_to_canonical(ref_seq, pred_feature_seqs[0])
    print("Aligned pred seq:    ", aligned)

    # Step 4: count
    counts = count_phonological_mdd(canonical, human, logits)
    summary = counts.summary()

    print("\n=== FINAL METRICS ===")
    print(summary["__macro_avg__"])

    print("============================================\n")