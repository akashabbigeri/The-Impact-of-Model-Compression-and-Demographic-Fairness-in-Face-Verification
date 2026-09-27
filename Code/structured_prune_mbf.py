"""
Structured (channel) pruning for MobileFaceNet (buffalo_s / w600k_mbf.onnx).
=============================================================================
MobileFaceNet uses MobileNetV2/V3-style inverted-residual blocks, NOT
ResNet's BN-Conv-PReLU-Conv-Add pattern used by structured_prune.py. Each
identity-shortcut block is:

    x -> Conv(expand, 1x1, groups=1) -> PReLU
      -> Conv(depthwise, 3x3, groups=C) -> PReLU
      -> Conv(project, 1x1, groups=1, linear)
      -> Add(x, project_out)

The expand-output / depthwise-channels / project-input are ONE shared width
(prunable together); the project's OUTPUT channel count is fixed by the Add
and must not change. The depthwise conv is special: it is a per-channel
(groups == channel count) conv, so pruning it means BOTH slicing its weight
(shape (C,1,kh,kw)) along dim0 AND updating its `group` attribute to the new
channel count — unlike a normal conv, channels can't be pruned by weight
slicing alone.

Stage-entry convs (stem, and the stride-2 downsampling convs between stages)
are deliberately left untouched: their output is the identity/shortcut base
for following blocks, so pruning them would require touching every
downstream Add, unlike the internal mid-width.

This mirrors structured_prune.py's approach (magnitude/L1 channel selection,
real dimension reduction, `block_indices` to restrict which blocks are
pruned) but adapted for this architecture's block pattern.
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRC_WEIGHTS = Path("~/.insightface/models/buffalo_s/w600k_mbf.onnx").expanduser()
OUT_DIR     = Path("weights_variants")
OUT_WEIGHTS = OUT_DIR / "w600k_mbf_pruned.onnx"

PRUNE_RATIO = 0.15  # fraction of shared mid-width removed, within pruned blocks

# Sensitivity sweep (mbf_prune_sensitivity.py + follow-up) on an 800-pair LFW
# subset, one-shot magnitude pruning (no fine-tuning available in this pipeline):
#   full-network (all 12 blocks): 10% -> 94.7%, 20% -> 72.4%, 30% -> 68.4%,
#     40% -> 64.9% (collapses fast, same mechanism as ResNet-50: project-conv
#     output feeds the Add with no norm layer between, error compounds)
#   last 4 of 12 blocks only: 5% -> 99.8%, 10% -> 99.6%, 15% -> 99.6%,
#     20% -> 99.4%, 25% -> 99.1%, 30% -> 98.1%, 35% -> 96.4%, 40% -> 94.0%
# Unlike ResNet-50 (which tolerated 40% on its last-3-blocks subset with
# ~0.3pp loss on the FULL held-out eval), MobileFaceNet's subset estimates
# were consistently optimistic vs. the full 10-fold held-out result here:
#   25% -> subset 99.12%, full held-out 98.81% (0.76pp drop, over tolerance)
#   20% -> subset 99.37%, full held-out 98.93% (0.64pp drop, still over)
# Both exceeded the ~0.5pp tolerance despite looking safe in-sample. Settled
# on 15% (last 4 blocks) as the final config -- the 40% target used for
# ResNet-50 does not transfer to this architecture at all; confirm the
# actual held-out number in baseline_mobilenetv3_pruned.json.
PRUNE_BLOCK_INDICES = {8, 9, 10, 11}


def _node_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
    return default


def find_prunable_quads(graph: onnx.GraphProto):
    """
    Return list of dicts describing each prunable inverted-residual block:
    {expand, depthwise, project, add} node references, in block order.
    Only matches identity-shortcut blocks (Add of raw input + project-out)
    with a clean, non-branching expand->PReLU->depthwise->PReLU->project chain.
    """
    producers = {}
    consumers = defaultdict(list)
    for n in graph.node:
        for out in n.output:
            producers[out] = n
        for inp in n.input:
            consumers[inp].append(n)

    def single_consumer(tensor_name):
        cs = consumers.get(tensor_name, [])
        return cs[0] if len(cs) == 1 else None

    quads = []
    for add_node in graph.node:
        if add_node.op_type != "Add":
            continue
        for side in (0, 1):
            project = producers.get(add_node.input[side])
            if project is None or project.op_type != "Conv":
                continue
            if _node_attr(project, "group", 1) != 1:
                continue
            if single_consumer(project.output[0]) is not add_node:
                continue  # project output must go ONLY to this Add

            prelu2 = producers.get(project.input[0])
            if prelu2 is None or prelu2.op_type != "PRelu":
                continue
            if single_consumer(prelu2.output[0]) is not project:
                continue

            depthwise = producers.get(prelu2.input[0])
            if depthwise is None or depthwise.op_type != "Conv":
                continue
            dw_group = _node_attr(depthwise, "group", 1)
            if dw_group == 1:
                continue  # not a depthwise conv
            if single_consumer(depthwise.output[0]) is not prelu2:
                continue

            prelu1 = producers.get(depthwise.input[0])
            if prelu1 is None or prelu1.op_type != "PRelu":
                continue
            if single_consumer(prelu1.output[0]) is not depthwise:
                continue

            expand = producers.get(prelu1.input[0])
            if expand is None or expand.op_type != "Conv":
                continue
            if _node_attr(expand, "group", 1) != 1:
                continue
            if single_consumer(expand.output[0]) is not prelu1:
                continue

            quads.append({
                "expand": expand, "prelu1": prelu1,
                "depthwise": depthwise, "prelu2": prelu2,
                "project": project, "add": add_node,
            })
            break  # found the main-path side; don't check the other side too

    return quads


def get_initializer(graph, name):
    for i, init in enumerate(graph.initializer):
        if init.name == name:
            return i, init
    return None, None


def replace_initializer(graph, name, arr):
    idx, _ = get_initializer(graph, name)
    if idx is None:
        raise KeyError(f"Initializer {name} not found in graph")
    new_init = numpy_helper.from_array(arr.astype(np.float32), name=name)
    graph.initializer.remove(graph.initializer[idx])
    graph.initializer.insert(idx, new_init)


def set_group_attr(node, value: int):
    for a in node.attribute:
        if a.name == "group":
            a.i = value
            return
    raise KeyError(f"Node {node.name or node.output[0]} has no 'group' attribute")


def prune_model(src_path: Path, out_path: Path, prune_ratio: float, block_indices=None) -> dict:
    if not src_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found at {src_path}")

    log.info(f"Loading model from {src_path}")
    model = onnx.load(str(src_path))
    graph = model.graph

    quads = find_prunable_quads(graph)
    if not quads:
        raise RuntimeError(
            "No prunable expand-depthwise-project blocks found. Graph topology "
            "does not match the expected MobileFaceNet inverted-residual pattern; "
            "refusing to prune blindly."
        )
    log.info(f"Found {len(quads)} prunable inverted-residual blocks")

    total_orig_params = 0
    total_pruned_params = 0
    total_channels_orig = 0
    total_channels_kept = 0
    n_pruned = 0

    for block_idx, q in enumerate(quads):
        if block_indices is not None and block_idx not in block_indices:
            log.info(f"  block {block_idx:2d}: skipped (not in block_indices)")
            continue

        expand, depthwise, project = q["expand"], q["depthwise"], q["project"]
        w_exp_name = expand.input[1]
        b_exp_name = expand.input[2] if len(expand.input) > 2 else None
        w_dw_name  = depthwise.input[1]
        b_dw_name  = depthwise.input[2] if len(depthwise.input) > 2 else None
        w_proj_name = project.input[1]

        _, w_exp_init = get_initializer(graph, w_exp_name)
        _, w_dw_init  = get_initializer(graph, w_dw_name)
        _, w_proj_init = get_initializer(graph, w_proj_name)
        if w_exp_init is None or w_dw_init is None or w_proj_init is None:
            raise RuntimeError(f"Missing initializer for block {block_idx}")

        w_exp = numpy_helper.to_array(w_exp_init).copy()   # (C, in_c, 1, 1)
        w_dw  = numpy_helper.to_array(w_dw_init).copy()    # (C, 1, kh, kw)  depthwise
        w_proj = numpy_helper.to_array(w_proj_init).copy() # (out_c, C, 1, 1)

        out_c = w_exp.shape[0]
        if w_dw.shape[0] != out_c or w_proj.shape[1] != out_c:
            raise RuntimeError(
                f"Shape mismatch at block {block_idx}: expand out_c={out_c}, "
                f"depthwise c={w_dw.shape[0]}, project in_c={w_proj.shape[1]}"
            )

        n_keep = max(1, round(out_c * (1 - prune_ratio)))
        l1 = np.abs(w_exp).sum(axis=(1, 2, 3))
        keep_idx = np.sort(np.argsort(-l1)[:n_keep])

        w_exp_new = w_exp[keep_idx]
        w_dw_new  = w_dw[keep_idx]
        w_proj_new = w_proj[:, keep_idx]

        # Project's output feeds the residual Add directly (no norm layer in
        # between in this architecture either), so rescale to compensate for
        # the reduced number of summed input channels -- same fix as ResNet-50.
        w_proj_new = w_proj_new * (out_c / n_keep)

        replace_initializer(graph, w_exp_name, w_exp_new)
        replace_initializer(graph, w_dw_name, w_dw_new)
        replace_initializer(graph, w_proj_name, w_proj_new)
        set_group_attr(depthwise, n_keep)

        if b_exp_name is not None:
            _, b_exp_init = get_initializer(graph, b_exp_name)
            if b_exp_init is not None:
                b_exp = numpy_helper.to_array(b_exp_init).copy()
                replace_initializer(graph, b_exp_name, b_exp[keep_idx])
        if b_dw_name is not None:
            _, b_dw_init = get_initializer(graph, b_dw_name)
            if b_dw_init is not None:
                b_dw = numpy_helper.to_array(b_dw_init).copy()
                replace_initializer(graph, b_dw_name, b_dw[keep_idx])

        # PReLU slope between expand and depthwise is per-channel too.
        prelu1_slope_name = q["prelu1"].input[1]
        _, slope_init = get_initializer(graph, prelu1_slope_name)
        if slope_init is not None:
            slope = numpy_helper.to_array(slope_init).copy()
            replace_initializer(graph, prelu1_slope_name, slope[keep_idx])

        # PReLU slope between depthwise and project is also per-channel,
        # over the SAME shared width.
        prelu2_slope_name = q["prelu2"].input[1]
        _, slope2_init = get_initializer(graph, prelu2_slope_name)
        if slope2_init is not None:
            slope2 = numpy_helper.to_array(slope2_init).copy()
            replace_initializer(graph, prelu2_slope_name, slope2[keep_idx])

        total_orig_params += w_exp.size + w_dw.size + w_proj.size
        total_pruned_params += w_exp_new.size + w_dw_new.size + w_proj_new.size
        total_channels_orig += out_c
        total_channels_kept += n_keep
        n_pruned += 1

        log.info(f"  block {block_idx:2d}: {out_c:4d} -> {n_keep:4d} channels kept "
                  f"({(out_c - n_keep) / out_c:.1%} pruned)")

    sparsity_by_channels = 1 - (total_channels_kept / total_channels_orig) if total_channels_orig else 0.0
    sparsity_by_params = 1 - (total_pruned_params / total_orig_params) if total_orig_params else 0.0
    total_removed_params = total_orig_params - total_pruned_params

    onnx.checker.check_model(model)

    OUT_DIR.mkdir(exist_ok=True)
    onnx.save(model, str(out_path))
    log.info(f"Saved pruned model to {out_path}")

    return {
        "n_blocks_total": len(quads),
        "n_blocks_pruned": n_pruned,
        "prune_ratio_requested": prune_ratio,
        "sparsity_achieved_by_channels": sparsity_by_channels,
        "sparsity_achieved_by_params_in_pruned_convs": sparsity_by_params,
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
