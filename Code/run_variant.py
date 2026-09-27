"""
Evaluation runner for a compressed model variant.
===================================================
Deliberately duplicates only the thin orchestration from baseline_pipeline.main()
(~load model, run 10-fold, compute CI/fairness/latency, dump JSON). Every actual
metric-computing function (lfw_10fold, compute_fairness_metrics, bootstrap_ci,
measure_latency, parse_pairs, load_attributes, preprocess/detector) is imported
from baseline_pipeline UNCHANGED, so results are directly comparable across
FP32/pruned/INT8 runs — only the model artefact passed to load_arcface differs.

Usage:
    python run_variant.py --weights weights_variants/w600k_r50_pruned.onnx \
        --model-name "ResNet-50 ArcFace Structured-Pruned (40% mid-channel sparsity)" \
        --out baseline_resnet50_pruned.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

import baseline_pipeline as bp

log = logging.getLogger(__name__)


def run(weights_path: Path, model_name: str, out_name: str) -> dict:
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    log.info(f"=== Variant Eval: {model_name} ===")
    session = bp.load_arcface(weights_path)

    pairs = bp.parse_pairs(bp.PAIRS_FILE)
    all_paths = [p for pair in pairs for p in pair[:2]]
    cache = bp.build_embedding_cache(session, all_paths)

    log.info("Running 10-fold evaluation...")
    results = bp.lfw_10fold(pairs, cache)
    log.info(f"Accuracy: {results['accuracy_mean']:.4f} +/- {results['accuracy_std']:.4f}")

    fpr, tpr, thresholds = roc_curve(results["all_labels"], results["all_scores"])
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_threshold = float(thresholds[eer_idx])
    log.info(f"EER threshold: {eer_threshold:.4f}")

    ci = bp.bootstrap_ci(results["all_scores"], results["all_labels"], eer_threshold)
    log.info(f"95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    attributes = bp.load_attributes(bp.ATTR_FILE)
    if attributes:
        log.info("Computing fairness metrics...")
        fairness = bp.compute_fairness_metrics(pairs, cache, attributes, eer_threshold)
    else:
        fairness = {}

    log.info("Measuring latency...")
    latency = bp.measure_latency(session)
    log.info(f"Latency: {latency['latency_median_ms']:.2f} ms (median)")

    output = {
        "model":         model_name,
        "weights_path":  str(weights_path),
        "accuracy_mean": results["accuracy_mean"],
        "accuracy_std":  results["accuracy_std"],
        "eer_threshold": eer_threshold,
        "bootstrap_ci":  ci,
        "fairness":      fairness,
        "latency":       latency,
    }

    out_path = bp.RESULTS_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved to {out_path}")
    return output


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        run(args.weights, args.model_name, args.out)
    except Exception as e:
        log.error(f"Variant evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
