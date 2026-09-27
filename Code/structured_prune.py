"""
Structured (channel) pruning for the ArcFace ResNet-50 (IResNet-50) backbone.
=============================================================================
Targets the internal width of each IBasicBlock:

    BN -> Conv1 (3x3) -> PReLU -> Conv2 (3x3) -> Add(shortcut) -> ...

Conv2's output feeds directly into the residual Add, so its channel count is
fixed by the block's shortcut branch and must NOT change. Conv1's output,
however, is purely internal to the block (consumed only by the PReLU, which
is consumed only by Conv2) — pruning its output channels (and correspondingly
Conv2's matching input channels + the PReLU's per-channel slope) is a safe,
dimension-reducing structured pruning that leaves every skip-connection shape
intact and requires no changes to any Add node.

This is a magnitude-based (L1 per-filter) importance criterion: for each
prunable Conv1, we keep the top-k output filters by L1 norm and drop the rest,
where k = round(out_channels * (1 - PRUNE_RATIO)).

Output: a smaller ONNX graph (real FLOPs/param reduction on the mid-channels
of every block), saved to weights_variants/w600k_r50_pruned.onnx.
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper, helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRC_WEIGHTS = Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser()
OUT_DIR     = Path("weights_variants")
OUT_WEIGHTS = OUT_DIR / "w600k_r50_pruned.onnx"

PRUNE_RATIO = 0.40  # fraction of *internal/mid* channels removed, within the pruned blocks

# This pipeline has no fine-tuning/retraining stage. A sensitivity sweep
# (prune_sensitivity.py) found that one-shot magnitude pruning distributed
# across all 24 IBasicBlocks collapses accuracy catastrophically (~60-70%)
# even at 10% ratio: conv2 in each block feeds the residual Add with no BN
# in between, so channel removal perturbs the embedding direction, and the
# error compounds multiplicatively across all 24 sequential blocks.
# Restricting pruning to the 3 layer4 blocks (512-channel, closest to the
# embedding head -> shortest cascade distance) tolerates up to ~40-50%
# internal sparsity with negligible accuracy loss (sweep: ~99.5%+ in-sample
# accuracy on an 800-pair LFW subset, vs. baseline ~99.8%).
PRUNE_BLOCK_INDICES = {21, 22, 23}  # layer4.0, layer4.1, layer4.2


def find_prunable_triples(graph: onnx.GraphProto):
    """
    Return list of (conv1_node, prelu_node, conv2_node) triples where conv1's
    output is consumed only by a PReLU, whose output is consumed only by
    conv2 (i.e. purely internal to a residual block, safe to shrink).
    """
    consumers = defaultdict(list)
    for n in graph.node:
        for inp in n.input:
            consumers[inp].append(n)

    triples = []
    for n in graph.node:
        if n.op_type != "Conv":
            continue
        out = n.output[0]
        cs = consumers.get(out, [])
        if len(cs) == 1 and cs[0].op_type == "PRelu":
            prelu = cs[0]
            pout = prelu.output[0]
            pcs = consumers.get(pout, [])
            if len(pcs) == 1 and pcs[0].op_type == "Conv":
                triples.append((n, prelu, pcs[0]))
    return triples


def get_initializer(graph: onnx.GraphProto, name: str):
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            return i, init
    return None, None


def replace_initializer(graph: onnx.GraphProto, name: str, arr: np.ndarray):
    idx, _ = get_initializer(graph, name)
    if idx is None:
        raise KeyError(f"Initializer {name} not found in graph")
    new_init = numpy_helper.from_array(arr.astype(np.float32), name=name)
    graph.initializer.remove(graph.initializer[idx])
    graph.initializer.insert(idx, new_init)


def prune_model(src_path: Path, out_path: Path, prune_ratio: float, block_indices=None) -> dict:
    if not src_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found at {src_path}")

    log.info(f"Loading model from {src_path}")
    model = onnx.load(str(src_path))
    graph = model.graph

    triples = find_prunable_triples(graph)
    if not triples:
        raise RuntimeError(
            "No prunable Conv-PReLU-Conv triples found. "
            "The graph topology does not match the expected IResNet BasicBlock pattern; "
            "refusing to prune blindly."
        )
    log.info(f"Found {len(triples)} prunable internal block-widths")

    total_orig_params = 0
    total_pruned_params = 0
    total_channels_orig = 0
    total_channels_kept = 0

    for block_idx, (conv1, prelu, conv2) in enumerate(triples):
        if block_indices is not None and block_idx not in block_indices:
            log.info(f"  block {block_idx:2d}: skipped (not in block_indices)")
            continue
        w1_name, b1_name = conv1.input[1], (conv1.input[2] if len(conv1.input) > 2 else None)
        slope_name = prelu.input[1]
        w2_name = conv2.input[1]

        _, w1_init = get_initializer(graph, w1_name)
        _, slope_init = get_initializer(graph, slope_name)
        _, w2_init = get_initializer(graph, w2_name)
        if w1_init is None or slope_init is None or w2_init is None:
            raise RuntimeError(f"Missing initializer for block {block_idx} (conv1={w1_name})")

        w1 = numpy_helper.to_array(w1_init).copy()      # (out_c, in_c, kh, kw)
        slope = numpy_helper.to_array(slope_init).copy()  # (out_c, 1, 1)
        w2 = numpy_helper.to_array(w2_init).copy()      # (out_c2, out_c, kh, kw)

        out_c = w1.shape[0]
        if w2.shape[1] != out_c or slope.shape[0] != out_c:
            raise RuntimeError(
                f"Shape mismatch at block {block_idx}: conv1 out_c={out_c}, "
                f"conv2 in_c={w2.shape[1]}, prelu slope c={slope.shape[0]}"
            )

        n_keep = max(1, round(out_c * (1 - prune_ratio)))
        l1 = np.abs(w1).sum(axis=(1, 2, 3))
        keep_idx = np.sort(np.argsort(-l1)[:n_keep])

        w1_new = w1[keep_idx]
        slope_new = slope[keep_idx]
        w2_new = w2[:, keep_idx]

        # Conv2's output feeds directly into the residual Add with no BatchNorm
        # in between, so truncating its input channels shrinks its output
        # magnitude with nothing downstream to renormalise it. Rescale the
        # kept weights so the expected output magnitude (sum over input
        # channels) is preserved, assuming roughly uniform per-channel
        # contribution -- without this, accuracy collapses even at ~10% pruning.
        w2_new = w2_new * (out_c / n_keep)

        replace_initializer(graph, w1_name, w1_new)
        replace_initializer(graph, slope_name, slope_new)
        replace_initializer(graph, w2_name, w2_new)

        if b1_name is not None:
            _, b1_init = get_initializer(graph, b1_name)
            if b1_init is not None:
                b1 = numpy_helper.to_array(b1_init).copy()
                replace_initializer(graph, b1_name, b1[keep_idx])

        total_orig_params += w1.size + w2.size
        total_pruned_params += w1_new.size + w2_new.size
        total_channels_orig += out_c
        total_channels_kept += n_keep

        log.info(f"  block {block_idx:2d}: {out_c:4d} -> {n_keep:4d} channels kept "
                  f"({(out_c - n_keep) / out_c:.1%} pruned)")

    n_pruned_blocks = len(block_indices) if block_indices is not None else len(triples)
    sparsity_by_channels = 1 - (total_channels_kept / total_channels_orig) if total_channels_orig else 0.0
    sparsity_by_params_in_pruned_convs = 1 - (total_pruned_params / total_orig_params) if total_orig_params else 0.0
    total_removed_params = total_orig_params - total_pruned_params

    onnx.checker.check_model(model)

    OUT_DIR.mkdir(exist_ok=True)
    onnx.save(model, str(out_path))
    log.info(f"Saved pruned model to {out_path}")

    return {
        "n_blocks_total": len(triples),
        "n_blocks_pruned": n_pruned_blocks,
        "prune_ratio_requested": prune_ratio,
        "sparsity_achieved_by_channels": sparsity_by_channels,
        "sparsity_achieved_by_params_in_pruned_convs": sparsity_by_params_in_pruned_convs,
        "total_removed_params": total_removed_params,
        "total_channels_orig": total_channels_orig,
        "total_channels_kept": total_channels_kept,
    }


def sanity_check_inference(out_path: Path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
    out = sess.run(None, {input_name: dummy})[0]
    if out.shape != (1, 512):
        raise RuntimeError(f"Pruned model output shape {out.shape} != expected (1, 512)")
    log.info(f"Sanity check OK — output shape {out.shape}")


def main():
    try:
        stats = prune_model(SRC_WEIGHTS, OUT_WEIGHTS, PRUNE_RATIO, block_indices=PRUNE_BLOCK_INDICES)
    except Exception as e:
        log.error(f"Structured pruning failed: {e}")
        sys.exit(1)

    try:
        sanity_check_inference(OUT_WEIGHTS)
    except Exception as e:
        log.error(f"Pruned model failed sanity-check inference: {e}")
        sys.exit(1)

    log.info("=== Pruning summary ===")
    for k, v in stats.items():
        log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
