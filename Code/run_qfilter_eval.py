"""
RQ3 step 3-4: quality-filtered re-evaluation of all 6 existing model variants.
================================================================================
Does NOT modify baseline_pipeline.py's eval logic. Quality filtering is applied
the same way baseline_pipeline already handles undetected faces: a pair is
excluded from lfw_10fold / compute_fairness_metrics whenever either image's
cache entry is None -- here we simply set cache entries below the quality
cutoff to None before calling those (unmodified) functions, so a pair is
"filtered" precisely when either partner is a low-quality image.

Embeddings are computed once per model from the cached SCRFD-aligned face112
crops (qfilter_detect_and_score.py), then reused across all 3 thresholds --
avoids re-running face detection or re-measuring latency 18 times.

Outputs one JSON per (model, threshold), e.g. results/baseline_resnet50_fp32_qfilt10.json,
following the same schema as the existing baseline_*.json files, plus a
qfilter_confound_check.json reporting per-subgroup removal rates.
"""
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

import baseline_pipeline as bp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("weights_variants/qfilter_cache")
QUALITY_JSON = Path("data/lfw_quality_scores.json")
RESULTS_DIR = bp.RESULTS_DIR
THRESHOLD_PCTS = [10, 20, 30]
MIN_VALID_PAIRS = 50

VARIANTS = [
    ("r50_fp32",   bp.WEIGHTS_PATH,                                               "ResNet-50 ArcFace FP32 (baseline)",                          "baseline_resnet50_fp32"),
    ("r50_pruned", Path("weights_variants/w600k_r50_pruned.onnx"),                "ResNet-50 ArcFace Structured-Pruned (layer4-only, 40% mid-channel sparsity)", "baseline_resnet50_pruned"),
    ("r50_int8",   Path("weights_variants/w600k_r50_int8.onnx"),                  "ResNet-50 ArcFace INT8 (static QDQ, per-channel weights)",   "baseline_resnet50_int8"),
    ("mbf_fp32",   Path("~/.insightface/models/buffalo_s/w600k_mbf.onnx").expanduser(), "MobileFaceNet ArcFace FP32 (buffalo_s, InsightFace MS1MV3)", "baseline_mobilenetv3_fp32"),
    ("mbf_pruned", Path("weights_variants/w600k_mbf_pruned.onnx"),                "MobileFaceNet ArcFace Structured-Pruned (last-4-of-12 blocks, 15% mid-width sparsity)", "baseline_mobilenetv3_pruned"),
    ("mbf_int8",   Path("weights_variants/w600k_mbf_int8.onnx"),                  "MobileFaceNet ArcFace INT8 (static QDQ, per-channel weights, depthwise excluded)", "baseline_mobilenetv3_int8"),
]


def load_quality_scores() -> dict[str, float | None]:
    if not QUALITY_JSON.exists():
        raise FileNotFoundError(
            f"{QUALITY_JSON} not found -- run qfilter_detect_and_score.py first."
        )
    with open(QUALITY_JSON) as f:
        return json.load(f)


def load_face112_cache() -> tuple[np.ndarray, list[str]]:
    stack_path = CACHE_DIR / "face112_stack.npy"
    paths_path = CACHE_DIR / "face112_paths.json"
    if not stack_path.exists() or not paths_path.exists():
        raise FileNotFoundError(
            f"Face cache not found at {CACHE_DIR} -- run qfilter_detect_and_score.py first."
        )
    stack = np.load(stack_path)
    with open(paths_path) as f:
        paths = json.load(f)
    if stack.shape[0] != len(paths):
        raise RuntimeError(f"Cache shape mismatch: {stack.shape[0]} faces vs {len(paths)} paths")
    return stack, paths


def compute_cutoffs(quality_scores: dict) -> dict[int, float]:
    scored = np.array([s for s in quality_scores.values() if s is not None])
    cutoffs = {pct: float(np.percentile(scored, pct)) for pct in THRESHOLD_PCTS}
    log.info(f"Quality cutoffs (bottom-N%% excluded): {cutoffs}")
    return cutoffs


def confound_check(pairs, quality_scores, attributes, cutoffs) -> dict:
    """Per-subgroup removal rate at each threshold, independent of model choice."""
    all_paths = sorted({p for pair in pairs for p in pair[:2]})
    subgroup_of = {p: bp.get_subgroup(attributes, p) for p in all_paths}

    subgroup_totals: dict[str, int] = {}
    for p in all_paths:
        subgroup_totals[subgroup_of[p]] = subgroup_totals.get(subgroup_of[p], 0) + 1

    detection_failed = {p for p in all_paths if quality_scores.get(p) is None}
    report = {"total_images_per_subgroup": subgroup_totals,
              "detection_failures_per_subgroup": {}, "thresholds": {}}
    for p in detection_failed:
        sg = subgroup_of[p]
        report["detection_failures_per_subgroup"][sg] = report["detection_failures_per_subgroup"].get(sg, 0) + 1

    for pct, cutoff in cutoffs.items():
        removed_per_sg: dict[str, int] = {}
        for p in all_paths:
            score = quality_scores.get(p)
            if score is None or score < cutoff:
                sg = subgroup_of[p]
                removed_per_sg[sg] = removed_per_sg.get(sg, 0) + 1
        removal_rate_per_sg = {
            sg: removed_per_sg.get(sg, 0) / subgroup_totals[sg] for sg in subgroup_totals
        }
        report["thresholds"][pct] = {
            "cutoff_score": cutoff,
            "removed_per_subgroup": removed_per_sg,
            "removal_rate_per_subgroup": removal_rate_per_sg,
        }
    return report


def eval_one(session, pairs, base_cache, attributes, quality_scores, cutoff, model_name, weights_path):
    filtered_cache = {
        p: (emb if (quality_scores.get(p) is not None and quality_scores[p] >= cutoff) else None)
        for p, emb in base_cache.items()
    }

    def score_pairs(pair_list):
        n = 0
        for p1, p2, _ in pair_list:
            if filtered_cache.get(p1) is not None and filtered_cache.get(p2) is not None:
                n += 1
        return n

    n_valid = score_pairs(pairs)
    if n_valid < MIN_VALID_PAIRS:
        raise RuntimeError(f"Only {n_valid} valid pairs remain after filtering (< {MIN_VALID_PAIRS} minimum) "
                             f"for {model_name} -- threshold too aggressive, skipping.")

    results = bp.lfw_10fold(pairs, filtered_cache)
    fpr, tpr, thresholds = roc_curve(results["all_labels"], results["all_scores"])
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_threshold = float(thresholds[eer_idx])

    ci = bp.bootstrap_ci(results["all_scores"], results["all_labels"], eer_threshold)
    fairness = bp.compute_fairness_metrics(pairs, filtered_cache, attributes, eer_threshold)

    return {
        "model": model_name,
        "weights_path": str(weights_path),
        "accuracy_mean": results["accuracy_mean"],
        "accuracy_std": results["accuracy_std"],
        "eer_threshold": eer_threshold,
        "bootstrap_ci": ci,
        "fairness": fairness,
        "n_pairs_total": len(pairs),
        "n_pairs_valid_after_filter": n_valid,
    }


def main():
    pairs = bp.parse_pairs(bp.PAIRS_FILE)
    quality_scores = load_quality_scores()
    face112_stack, face112_paths = load_face112_cache()
    attributes = bp.load_attributes(bp.ATTR_FILE)
    cutoffs = compute_cutoffs(quality_scores)

    report = confound_check(pairs, quality_scores, attributes, cutoffs)
    with open(RESULTS_DIR / "qfilter_confound_check.json", "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Confound check saved to {RESULTS_DIR / 'qfilter_confound_check.json'}")
    for pct, info in report["thresholds"].items():
        log.info(f"  bottom-{pct}%: removal rate per subgroup: "
                  f"{ {k: f'{v:.1%}' for k, v in info['removal_rate_per_subgroup'].items()} }")

    for tag, weights_path, model_name, base_name in VARIANTS:
        log.info(f"=== {model_name} ===")
        if not Path(weights_path).exists():
            log.error(f"Weights not found at {weights_path} -- skipping {tag}")
            continue

        try:
            session = bp.load_arcface(Path(weights_path))
        except Exception as e:
            log.error(f"Failed to load {tag}: {e}")
            continue

        base_cache = {}
        for path, face in zip(face112_paths, face112_stack):
            try:
                base_cache[path] = bp.get_embedding(session, face[np.newaxis, ...])
            except Exception as e:
                log.warning(f"Embedding failed for {path} on {tag}: {e}")
                base_cache[path] = None

        existing_result_path = RESULTS_DIR / f"{base_name}.json"
        latency_block = None
        if existing_result_path.exists():
            with open(existing_result_path) as f:
                latency_block = json.load(f).get("latency")

        for pct in THRESHOLD_PCTS:
            cutoff = cutoffs[pct]
            out_path = RESULTS_DIR / f"{base_name}_qfilt{pct}.json"
            try:
                result = eval_one(session, pairs, base_cache, attributes, quality_scores,
                                    cutoff, model_name, weights_path)
            except RuntimeError as e:
                log.error(str(e))
                continue
            result["quality_filter"] = {
                "threshold_pct_excluded": pct,
                "quality_cutoff_score": cutoff,
                "note": "Pairs excluded if either image's FaceQNet quality score is below the "
                        "bottom-N% cutoff (computed on the full scored LFW population) or if "
                        "SCRFD detection failed on that image. Filtering applied via the same "
                        "None-cache-entry mechanism baseline_pipeline.py already uses for "
                        "undetected faces -- lfw_10fold/compute_fairness_metrics unmodified.",
            }
            if latency_block is not None:
                result["latency"] = latency_block
                result["latency_note"] = ("Reused from the unfiltered baseline run for this model -- "
                                            "latency is a per-inference model property independent of "
                                            "which pairs are included in the accuracy/fairness stats.")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            log.info(f"  qfilt{pct}: accuracy={result['accuracy_mean']:.4f} "
                      f"n_valid_pairs={result['n_pairs_valid_after_filter']}/{result['n_pairs_total']} "
                      f"FDR_FMR={result['fairness'].get('FDR_FMR', float('nan')):.4f} "
                      f"FDR_FNMR={result['fairness'].get('FDR_FNMR', float('nan')):.4f} "
                      f"-> {out_path}")

        del base_cache, session


if __name__ == "__main__":
    main()
