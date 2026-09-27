"""
Quick prune-ratio sensitivity sweep on a subset of LFW pairs (no fine-tuning
in this pipeline, so pruning is one-shot magnitude pruning — this sweep finds
a ratio that survives one-shot pruning before committing to a full 10-fold run).
Not part of the reported eval pipeline; diagnostic only.
"""
import logging
import numpy as np
from pathlib import Path

import structured_prune as sp
import baseline_pipeline as bp

logging.basicConfig(level=logging.WARNING)

N_SUBSET_PAIRS = 800

pairs = bp.parse_pairs(bp.PAIRS_FILE)
rng = np.random.default_rng(0)
idx = rng.choice(len(pairs), size=min(N_SUBSET_PAIRS, len(pairs)), replace=False)
subset = [pairs[i] for i in idx]
paths = list({p for pair in subset for p in pair[:2]})

import sys
LATE_ONLY = "--late-only" in sys.argv
# triples[21,22,23] are the three 512-channel blocks in layer4 (closest to the
# embedding head -> shortest cascade distance for compounding error).
LATE_BLOCKS = {21, 22, 23}

configs = [0.10, 0.20, 0.30, 0.40, 0.50] if LATE_ONLY else [0.10, 0.15, 0.20, 0.25, 0.30]

for ratio in configs:
    tmp_out = Path("weights_variants") / f"_sweep_{int(ratio*100)}.onnx"
    block_indices = LATE_BLOCKS if LATE_ONLY else None
    stats = sp.prune_model(sp.SRC_WEIGHTS, tmp_out, ratio, block_indices=block_indices)
    sess = bp.load_arcface(tmp_out)
    cache = {}
    for p in paths:
        img = __import__("cv2").imread(p)
        if img is None:
            cache[p] = None
            continue
        face = bp.preprocess(img)
        cache[p] = None if face is None else bp.get_embedding(sess, face)

    preds, labels = [], []
    for p1, p2, label in subset:
        e1, e2 = cache.get(p1), cache.get(p2)
        if e1 is None or e2 is None:
            continue
        preds.append(bp.cosine_similarity(e1, e2))
        labels.append(label)
    preds, labels = np.array(preds), np.array(labels)

    best_acc = 0.0
    for t in np.linspace(-1, 1, 200):
        acc = np.mean((preds >= t).astype(int) == labels)
        best_acc = max(best_acc, acc)

    print(f"ratio={ratio:.2f}  channel_sparsity={stats['sparsity_achieved_by_channels']:.3f}  "
          f"subset_best_acc={best_acc:.4f}")
    tmp_out.unlink(missing_ok=True)
