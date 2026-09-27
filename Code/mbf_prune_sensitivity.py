"""
Prune-ratio / block-subset sensitivity sweep for MobileFaceNet, on a subset
of LFW pairs (no fine-tuning in this pipeline, so pruning is one-shot).
Diagnostic only, mirrors prune_sensitivity.py's approach for ResNet-50.
"""
import sys
import logging
import numpy as np
from pathlib import Path

import structured_prune_mbf as sp
import baseline_pipeline as bp

logging.basicConfig(level=logging.WARNING)

N_SUBSET_PAIRS = 800
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"  # all | late | early

pairs = bp.parse_pairs(bp.PAIRS_FILE)
rng = np.random.default_rng(0)
idx = rng.choice(len(pairs), size=min(N_SUBSET_PAIRS, len(pairs)), replace=False)
subset = [pairs[i] for i in idx]
paths = list({p for pair in subset for p in pair[:2]})

N_BLOCKS = 12
if MODE == "all":
    ratios_and_blocks = [(r, None) for r in [0.10, 0.20, 0.30, 0.40]]
elif MODE == "late":
    late = set(range(8, 12))  # last 4 of 12 blocks
    ratios_and_blocks = [(r, late) for r in [0.20, 0.30, 0.40, 0.50]]
elif MODE == "early":
    early = set(range(0, 4))  # first 4 of 12 blocks
    ratios_and_blocks = [(r, early) for r in [0.20, 0.30, 0.40]]
else:
    raise ValueError(MODE)

for ratio, block_indices in ratios_and_blocks:
    tag = f"{MODE}_{int(ratio*100)}"
    tmp_out = Path("weights_variants") / f"_sweep_mbf_{tag}.onnx"
    stats = sp.prune_model(sp.SRC_WEIGHTS, tmp_out, ratio, block_indices=block_indices)
    sess = bp.load_arcface(tmp_out)

    import cv2
    cache = {}
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            cache[p] = None
            continue
        face = bp.preprocess(img)
        cache[p] = None if face is None else bp.get_embedding(sess, face)

    scores, labels = [], []
    for p1, p2, label in subset:
        e1, e2 = cache.get(p1), cache.get(p2)
        if e1 is None or e2 is None:
            continue
        scores.append(bp.cosine_similarity(e1, e2))
        labels.append(label)
    scores, labels = np.array(scores), np.array(labels)

    best_acc = 0.0
    for t in np.linspace(-1, 1, 200):
        acc = np.mean((scores >= t).astype(int) == labels)
        best_acc = max(best_acc, acc)

    print(f"mode={MODE:6s} ratio={ratio:.2f}  channel_sparsity={stats['sparsity_achieved_by_channels']:.3f}  "
          f"subset_best_acc={best_acc:.4f}")
    tmp_out.unlink(missing_ok=True)
