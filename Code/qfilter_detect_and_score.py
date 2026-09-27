"""
RQ3 step 1-2: FaceQNet-based quality scoring of LFW images.
=============================================================
Runs SCRFD detection ONCE per unique image referenced in pairs.txt (reusing
baseline_pipeline.py's already-loaded detector, unmodified). From each
detection we derive:
  - a 112x112 ArcFace-ready crop (identical maths to baseline_pipeline.preprocess,
    cached to disk so the 6 model-variant embedding passes never re-run detection)
  - a 224x224 BGR crop for FaceQNet (same 5-point alignment, insightface's
    norm_crop scales its template automatically for image_size=224)

FaceQNet (v1, official pretrained weights from uam-biometrics/FaceQnet) scores
each 224x224 crop in batches. Output: 0-1 quality score, higher = better
(per the authors' own postprocessing, which we replicate exactly: clip to [0,1]).

Outputs:
    data/lfw_quality_scores.json       -- {image_path: score or null}
    weights_variants/qfilter_cache/face112_stack.npy   -- (N, 1, 3, 112, 112) float32
    weights_variants/qfilter_cache/face112_paths.json  -- ordered path list matching the stack
    data/qfilter_detection_failures.json -- paths where SCRFD found no face (excluded everywhere)

Does NOT modify baseline_pipeline.py.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
import logging
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import baseline_pipeline as bp
from insightface.utils import face_align

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FACEQNET_H5 = Path("weights_variants/faceqnet/FaceQnet_v1.h5")
OUT_QUALITY_JSON = Path("data/lfw_quality_scores.json")
OUT_FAILURES_JSON = Path("data/qfilter_detection_failures.json")
CACHE_DIR = Path("weights_variants/qfilter_cache")
FACEQNET_BATCH = 32


def to_arcface_input(aligned_bgr_112: np.ndarray) -> np.ndarray:
    """Identical maths to baseline_pipeline.preprocess()'s post-alignment steps."""
    aligned_rgb = cv2.cvtColor(aligned_bgr_112, cv2.COLOR_BGR2RGB)
    face = aligned_rgb.transpose(2, 0, 1).astype(np.float32)
    face = face / 127.5 - 1.0
    return face[np.newaxis, ...]


def load_faceqnet():
    if not FACEQNET_H5.exists():
        raise FileNotFoundError(
            f"FaceQNet weights not found at {FACEQNET_H5}. Download from "
            "https://github.com/uam-biometrics/FaceQnet/releases/download/v1.0/FaceQnet_v1.h5"
        )
    import tf_keras
    model = tf_keras.models.load_model(str(FACEQNET_H5), compile=False)
    out_shape = model.output_shape
    if out_shape != (None, 1):
        raise RuntimeError(f"Unexpected FaceQNet output shape {out_shape}, expected (None, 1)")
    return model


def main():
    pairs = bp.parse_pairs(bp.PAIRS_FILE)
    all_paths = sorted({p for pair in pairs for p in pair[:2]})
    log.info(f"{len(all_paths)} unique images referenced in {bp.PAIRS_FILE}")

    faceqnet = load_faceqnet()
    log.info("FaceQNet v1 loaded")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    face112_list = []
    face112_paths = []
    quality_scores = {}
    failures = []

    batch_224 = []
    batch_paths = []

    def flush_batch():
        if not batch_224:
            return
        arr = np.stack(batch_224, axis=0).astype(np.float32)  # (B, 224, 224, 3) BGR raw
        preds = faceqnet.predict(arr, batch_size=len(batch_224), verbose=0)
        for p, s in zip(batch_paths, preds):
            score = float(np.clip(s[0], 0.0, 1.0))
            quality_scores[p] = score
        batch_224.clear()
        batch_paths.clear()

    for path in tqdm(all_paths, desc="Detecting + scoring quality"):
        img = cv2.imread(path)
        if img is None:
            failures.append(path)
            quality_scores[path] = None
            continue
        try:
            bboxes, kpss = bp.detector.detect(img, max_num=1, metric="default")
        except Exception as e:
            log.warning(f"Detector error on {path}: {e}")
            bboxes, kpss = None, None

        if bboxes is None or bboxes.shape[0] == 0 or kpss is None:
            failures.append(path)
            quality_scores[path] = None
            continue

        aligned_112 = face_align.norm_crop(img, kpss[0], image_size=112)
        face112_list.append(to_arcface_input(aligned_112))
        face112_paths.append(path)

        aligned_224 = face_align.norm_crop(img, kpss[0], image_size=224)
        batch_224.append(aligned_224)
        batch_paths.append(path)
        if len(batch_224) >= FACEQNET_BATCH:
            flush_batch()

    flush_batch()

    log.info(f"Detection succeeded for {len(face112_paths)}/{len(all_paths)} images "
              f"({len(failures)} failures)")

    if not face112_paths:
        raise RuntimeError("No faces detected in any image -- cannot proceed.")

    face112_stack = np.concatenate(face112_list, axis=0)  # (N, 1, 3, 112, 112)
    np.save(CACHE_DIR / "face112_stack.npy", face112_stack)
    with open(CACHE_DIR / "face112_paths.json", "w") as f:
        json.dump(face112_paths, f)
    log.info(f"Saved ArcFace-ready face cache: {face112_stack.shape} -> {CACHE_DIR / 'face112_stack.npy'}")

    with open(OUT_QUALITY_JSON, "w") as f:
        json.dump(quality_scores, f, indent=2)
    log.info(f"Saved quality scores for {len(quality_scores)} images -> {OUT_QUALITY_JSON}")

    with open(OUT_FAILURES_JSON, "w") as f:
        json.dump(failures, f, indent=2)
    if failures:
        log.warning(f"{len(failures)} images had no detected face -- excluded from all filtering "
                     f"thresholds and from ArcFace embedding cache. See {OUT_FAILURES_JSON}")

    scored = [s for s in quality_scores.values() if s is not None]
    scored = np.array(scored)
    log.info(f"Quality score distribution: min={scored.min():.4f} p10={np.percentile(scored,10):.4f} "
              f"p20={np.percentile(scored,20):.4f} p30={np.percentile(scored,30):.4f} "
              f"median={np.median(scored):.4f} max={scored.max():.4f}")


if __name__ == "__main__":
    main()
