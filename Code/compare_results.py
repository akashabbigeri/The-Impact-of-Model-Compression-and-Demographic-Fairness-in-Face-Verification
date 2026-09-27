"""
Summary comparison of FP32 / structured-pruned / INT8 quantised ArcFace
ResNet-50 results, all produced by the same unmodified eval logic in
baseline_pipeline.py (only the model artefact differs between runs).
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
ACCURACY_SANITY_THRESHOLD = 0.99  # flag pruning if accuracy drops below this

VARIANTS = [
    ("ResNet-50 FP32 (baseline)", "baseline_resnet50_fp32.json"),
    ("ResNet-50 Structured-Pruned", "baseline_resnet50_pruned.json"),
    ("ResNet-50 INT8 Quantised", "baseline_resnet50_int8.json"),
    ("MobileFaceNet FP32 (baseline)", "baseline_mobilenetv3_fp32.json"),
    ("MobileFaceNet Structured-Pruned", "baseline_mobilenetv3_pruned.json"),
    ("MobileFaceNet INT8 Quantised", "baseline_mobilenetv3_int8.json"),
]

# RQ3: FaceQNet quality-filtering sweep. (label, results-dir basename without .json)
RQ3_MODELS = [
    ("ResNet-50 FP32",       "baseline_resnet50_fp32"),
    ("ResNet-50 Pruned",     "baseline_resnet50_pruned"),
    ("ResNet-50 INT8",       "baseline_resnet50_int8"),
    ("MobileFaceNet FP32",   "baseline_mobilenetv3_fp32"),
    ("MobileFaceNet Pruned", "baseline_mobilenetv3_pruned"),
    ("MobileFaceNet INT8",   "baseline_mobilenetv3_int8"),
]
RQ3_LEVELS = [("none", ""), ("bot10", "_qfilt10"), ("bot20", "_qfilt20"), ("bot30", "_qfilt30")]
# Relative shrink in FDR_FNMR (none -> bot30) beyond which a model is called
# "quality-driven" rather than "inconclusive"/"flat". Chosen well above the ~40-45%
# shrink common to the uncompressed FP32 baselines themselves (a dataset-level
# quality/subgroup-size effect, not evidence of compression-specific masking).
QUALITY_DRIVEN_SHRINK_THRESHOLD = 0.70
FLAT_OR_GROWING_THRESHOLD = 0.0  # non-negative relative change


def load(path: Path) -> dict | None:
    if not path.exists():
        log.warning(f"Missing results file: {path} (skipping)")
        return None
    with open(path) as f:
        return json.load(f)


def classify_trend(fdr_none, fdr_bot30):
    """
    Classify a model's fairness-gap-vs-filtering trend. Deliberately conservative:
    with subgroup sizes this small (Asian n~150-320, Black n~130-260, Unknown n~7-35),
    single-pair differences move FDR by a lot, so only a large, consistent shrink is
    called "quality-driven" -- anything else is "flat/grows" or "inconclusive".
    """
    if fdr_none is None or fdr_bot30 is None:
        return "missing data"
    if fdr_none == 0:
        return "inconclusive (baseline already 0)"
    rel_change = (fdr_bot30 - fdr_none) / fdr_none
    if rel_change <= -QUALITY_DRIVEN_SHRINK_THRESHOLD:
        return f"quality-driven (shrinks {rel_change:+.0%})"
    if rel_change >= FLAT_OR_GROWING_THRESHOLD:
        return f"flat/grows ({rel_change:+.0%}) -- gap persists despite filtering"
    return f"inconclusive ({rel_change:+.0%}, within small-sample noise)"


def print_rq3_tables():
    data = {}
    for label, base in RQ3_MODELS:
        data[label] = {}
        for lvl, suffix in RQ3_LEVELS:
            data[label][lvl] = load(RESULTS_DIR / f"{base}{suffix}.json")

    def fmt(v):
        return f"{v:.4f}" if v is not None else "n/a"

    log.info("\n=== RQ3: FaceQNet quality filtering -- FDR_FMR ===")
    log.info(f"{'Model':<22} {'none':>8} {'bot10%':>8} {'bot20%':>8} {'bot30%':>8}")
    for label, _ in RQ3_MODELS:
        vals = [data[label][lvl]["fairness"]["FDR_FMR"] if data[label][lvl] else None
                for lvl, _ in RQ3_LEVELS]
        log.info(f"{label:<22} " + " ".join(fmt(v).rjust(8) for v in vals))

    log.info("\n=== RQ3: FaceQNet quality filtering -- FDR_FNMR ===")
    log.info(f"{'Model':<22} {'none':>8} {'bot10%':>8} {'bot20%':>8} {'bot30%':>8}  Trend")
    for label, _ in RQ3_MODELS:
        vals = [data[label][lvl]["fairness"]["FDR_FNMR"] if data[label][lvl] else None
                for lvl, _ in RQ3_LEVELS]
        trend = classify_trend(vals[0], vals[-1])
        log.info(f"{label:<22} " + " ".join(fmt(v).rjust(8) for v in vals) + f"  {trend}")

    log.info("\n=== RQ3: Asian subgroup FMR / FNMR (n_impostor / n_genuine) ===")
    for label, _ in RQ3_MODELS:
        log.info(f"\n{label}:")
        for lvl, _ in RQ3_LEVELS:
            d = data[label][lvl]
            if d is None:
                log.info(f"  {lvl:<6}: n/a")
                continue
            a = d["fairness"].get("Asian", {})
            log.info(f"  {lvl:<6}: FMR={fmt(a.get('FMR'))}  FNMR={fmt(a.get('FNMR'))}  "
                      f"n_impostor={a.get('n_impostor')}  n_genuine={a.get('n_genuine')}")

    confound_path = RESULTS_DIR / "qfilter_confound_check.json"
    if confound_path.exists():
        with open(confound_path) as f:
            confound = json.load(f)
        log.info("\n=== RQ3: subgroup removal-rate confound check ===")
        for pct, info in confound.get("thresholds", {}).items():
            rates = ", ".join(f"{sg}={rate:.1%}" for sg, rate in info["removal_rate_per_subgroup"].items())
            log.info(f"  bottom-{pct}%: {rates}")
        log.info("  NOTE: 'Unknown' subgroup is removed at ~3-4x the rate of White/Black/Asian "
                  "at every threshold -- treat Unknown-subgroup fairness swings as a filtering "
                  "confound, not a clean result.")
    else:
        log.warning(f"Confound check file not found at {confound_path}")


def main():
    rows = []
    for label, filename in VARIANTS:
        data = load(RESULTS_DIR / filename)
        if data is None:
            continue
        fairness = data.get("fairness", {})
        rows.append({
            "label":        label,
            "accuracy":     data.get("accuracy_mean"),
            "fdr_fmr":      fairness.get("FDR_FMR"),
            "fdr_fnmr":     fairness.get("FDR_FNMR"),
            "latency_ms":   data.get("latency", {}).get("latency_median_ms"),
        })

    if not rows:
        log.error("No result files found — nothing to compare.")
        return

    header = f"{'Model':<22} {'Accuracy':>10} {'FDR_FMR':>10} {'FDR_FNMR':>10} {'Latency(ms)':>12}"
    log.info(header)
    log.info("-" * len(header))
    for r in rows:
        acc = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "n/a"
        fmr = f"{r['fdr_fmr']:.4f}" if r['fdr_fmr'] is not None else "n/a"
        fnmr = f"{r['fdr_fnmr']:.4f}" if r['fdr_fnmr'] is not None else "n/a"
        lat = f"{r['latency_ms']:.2f}" if r['latency_ms'] is not None else "n/a"
        log.info(f"{r['label']:<22} {acc:>10} {fmr:>10} {fnmr:>10} {lat:>12}")

    for pruned in (r for r in rows if "Structured-Pruned" in r["label"]):
        if pruned["accuracy"] is not None and pruned["accuracy"] < ACCURACY_SANITY_THRESHOLD:
            log.warning(
                f"\n⚠ {pruned['label']} accuracy {pruned['accuracy']:.4f} is below the "
                f"{ACCURACY_SANITY_THRESHOLD:.2%} sanity threshold — consider reducing "
                f"the prune ratio."
            )

    print_rq3_tables()


if __name__ == "__main__":
    main()
