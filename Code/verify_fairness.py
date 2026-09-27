"""
Diagnostic: verify the ResNet-50 vs MobileFaceNet fairness numbers are
genuinely independently computed, not a caching/reuse artefact.

Scoped to only the images actually needed (spot-check pairs + full Asian
subgroup) rather than the full 7701-image LFW set, so this runs in minutes.
Uses the EER thresholds already produced by the two independent production
runs (baseline_resnet50_fp32.json / baseline_mobilenetv3_fp32.json) rather
than recomputing full 10-fold + bootstrap, since we're checking the fairness
computation and score independence, not re-deriving the threshold.
"""
import json
import logging
import numpy as np

import baseline_pipeline as bp

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

RESNET_WEIGHTS = bp.WEIGHTS_PATH
MOBILE_WEIGHTS = bp.Path("~/.insightface/models/buffalo_s/w600k_mbf.onnx").expanduser()

with open(bp.RESULTS_DIR / "baseline_resnet50_fp32.json") as f:
    thr_r50 = json.load(f)["eer_threshold"]
with open(bp.RESULTS_DIR / "baseline_mobilenetv3_fp32.json") as f:
    thr_mbf = json.load(f)["eer_threshold"]

log.info(f"Thresholds from production JSONs: ResNet50={thr_r50:.6f}  MobileFaceNet={thr_mbf:.6f}")
log.info(f"Thresholds differ: {thr_r50 != thr_mbf}")

pairs = bp.parse_pairs(bp.PAIRS_FILE)
attributes = bp.load_attributes(bp.ATTR_FILE)

asian_pairs = [(p1, p2, label) for p1, p2, label in pairs
               if bp.get_subgroup(attributes, p1) == "Asian"]
log.info(f"Asian-subgroup pairs: {len(asian_pairs)}")

spot_pairs = pairs[:10]
needed_paths = sorted({p for pair in (asian_pairs + spot_pairs) for p in pair[:2]})
log.info(f"Total unique images needed for this diagnostic: {len(needed_paths)}")

log.info("Loading both sessions fresh in this single process...")
sess_r50 = bp.load_arcface(RESNET_WEIGHTS)
sess_mbf = bp.load_arcface(MOBILE_WEIGHTS)

log.info("Building scoped embedding caches (independent objects)...")
cache_r50 = bp.build_embedding_cache(sess_r50, needed_paths)
cache_mbf = bp.build_embedding_cache(sess_mbf, needed_paths)

assert cache_r50 is not cache_mbf, "BUG: caches are the same object"

# --- Step 2: spot-check raw scores on first 10 pairs ---
log.info("\n=== Spot-check: raw cosine similarity, first 10 pairs ===")
log.info(f"{'label':>5} {'ResNet50':>12} {'MobileFaceNet':>14} {'diff':>10}")
for p1, p2, label in spot_pairs:
    e1r, e2r = cache_r50.get(p1), cache_r50.get(p2)
    e1m, e2m = cache_mbf.get(p1), cache_mbf.get(p2)
    if any(x is None for x in (e1r, e2r, e1m, e2m)):
        continue
    sr = bp.cosine_similarity(e1r, e2r)
    sm = bp.cosine_similarity(e1m, e2m)
    log.info(f"{label:>5} {sr:>12.6f} {sm:>14.6f} {sr - sm:>10.6f}")

sample_path = needed_paths[0]
e_r = cache_r50[sample_path]
e_m = cache_mbf[sample_path]
log.info(f"\nSame image embedding, ResNet-50 first 5 dims:      {e_r[:5]}")
log.info(f"Same image embedding, MobileFaceNet first 5 dims:  {e_m[:5]}")
log.info(f"Embeddings identical object? {e_r is e_m}")
log.info(f"Embeddings numerically equal? {np.allclose(e_r, e_m)}")

# --- Step 3 + 5: recompute Asian-subgroup fairness fresh with each model's
#     own threshold, and identify exactly which pairs are misclassified ---
def asian_fairness_and_misclassified(cache, threshold):
    imp, gen = [], []
    fp_pairs, fn_pairs = [], []
    for p1, p2, label in asian_pairs:
        e1, e2 = cache.get(p1), cache.get(p2)
        if e1 is None or e2 is None:
            continue
        score = bp.cosine_similarity(e1, e2)
        pred = int(score >= threshold)
        if label == 0:
            imp.append(score)
            if pred == 1:
                fp_pairs.append((p1, p2, score))
        else:
            gen.append(score)
            if pred == 0:
                fn_pairs.append((p1, p2, score))
    imp, gen = np.array(imp), np.array(gen)
    fmr = float(np.mean(imp >= threshold)) if len(imp) else None
    fnmr = float(np.mean(gen < threshold)) if len(gen) else None
    return fmr, fnmr, fp_pairs, fn_pairs, len(imp), len(gen)

fmr_r50, fnmr_r50, fp_r50, fn_r50, n_imp_r50, n_gen_r50 = asian_fairness_and_misclassified(cache_r50, thr_r50)
fmr_mbf, fnmr_mbf, fp_mbf, fn_mbf, n_imp_mbf, n_gen_mbf = asian_fairness_and_misclassified(cache_mbf, thr_mbf)

log.info(f"\n=== Asian subgroup, recomputed fresh in this process ===")
log.info(f"ResNet50:      FMR={fmr_r50} ({len(fp_r50)}/{n_imp_r50})  FNMR={fnmr_r50} ({len(fn_r50)}/{n_gen_r50})")
log.info(f"MobileFaceNet: FMR={fmr_mbf} ({len(fp_mbf)}/{n_imp_mbf})  FNMR={fnmr_mbf} ({len(fn_mbf)}/{n_gen_mbf})")

log.info(f"\nResNet50      false-match pairs: {[(p1, p2) for p1,p2,_ in fp_r50]}")
log.info(f"MobileFaceNet false-match pairs: {[(p1, p2) for p1,p2,_ in fp_mbf]}")
log.info(f"ResNet50      false-nonmatch pairs: {[(p1, p2) for p1,p2,_ in fn_r50]}")
log.info(f"MobileFaceNet false-nonmatch pairs: {[(p1, p2) for p1,p2,_ in fn_mbf]}")

same_fp = {(p1, p2) for p1, p2, _ in fp_r50} == {(p1, p2) for p1, p2, _ in fp_mbf}
same_fn = {(p1, p2) for p1, p2, _ in fn_r50} == {(p1, p2) for p1, p2, _ in fn_mbf}
log.info(f"\nSame false-match pairs across models? {same_fp}")
log.info(f"Same false-nonmatch pairs across models? {same_fn}")
