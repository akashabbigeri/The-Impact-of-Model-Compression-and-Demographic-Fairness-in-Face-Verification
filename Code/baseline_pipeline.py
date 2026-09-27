"""
Baseline Face Verification Pipeline
====================================
MTCNN detection → ArcFace ResNet-50 → cosine similarity → LFW 10-fold protocol

Requirements:
    pip install facenet-pytorch insightface onnxruntime opencv-python tqdm scikit-learn numpy

Pretrained ArcFace ResNet-50 weights:
    Download from InsightFace model zoo:
    https://github.com/deepinsight/insightface/tree/master/model_zoo
    Use: buffalo_l (ResNet-50, ArcFace, MS1MV3)
    Place .onnx file at: weights/arcface_r50.onnx

LFW dataset:
    Download from: http://vis-www.cs.umass.edu/lfw/lfw.tgz
    Pairs file: http://vis-www.cs.umass.edu/lfw/pairs.txt
    Place images at: data/lfw/
    Place pairs.txt at: data/pairs.txt

LFW Attributes (demographic labels):
    Download from: https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
    Place at: data/lfw_attributes.txt
"""

import os
import csv
import logging
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve
from insightface.model_zoo import model_zoo
from insightface.utils import face_align
import onnxruntime as ort
import torch

# ── Config ────────────────────────────────────────────────────────────────────
LFW_DIR        = Path("data/lfw")
PAIRS_FILE     = Path("data/pairs.txt")
ATTR_FILE      = Path("data/lfw_attributes.txt")
WEIGHTS_PATH = Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser()
DET_WEIGHTS_PATH = Path("~/.insightface/models/buffalo_l/det_10g.onnx").expanduser()
IMG_SIZE       = (112, 112)
RESULTS_DIR    = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Detector (InsightFace SCRFD, 5-point landmarks for ArcFace alignment) ─────
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Using device: {device}")

detector = model_zoo.get_model(str(DET_WEIGHTS_PATH))
detector.prepare(ctx_id=0 if device == "cuda" else -1, input_size=(640, 640))

# ── ArcFace ONNX model ────────────────────────────────────────────────────────
def load_arcface(weights_path: Path) -> ort.InferenceSession:
    """Load ArcFace ONNX model for inference."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"ArcFace weights not found at {weights_path}. "
            "Download from InsightFace model zoo: "
            "https://github.com/deepinsight/insightface/tree/master/model_zoo"
        )
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(weights_path), providers=providers)
    log.info(f"Loaded ArcFace model from {weights_path}")
    return session


# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect face with SCRFD, align to 112x112 using the 5-point ArcFace
    template (matches the alignment ArcFace was trained/benchmarked with),
    normalise to [-1, 1]. Returns None if no face detected.
    """
    try:
        bboxes, kpss = detector.detect(img_bgr, max_num=1, metric="default")
    except Exception as e:
        log.warning(f"Detector error: {e}")
        return None

    if bboxes.shape[0] == 0 or kpss is None:
        return None

    aligned_bgr = face_align.norm_crop(img_bgr, kpss[0], image_size=112)
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    face = aligned_rgb.transpose(2, 0, 1).astype(np.float32)  # (3, 112, 112), [0, 255]
    face = face / 127.5 - 1.0                                  # normalise to [-1, 1]
    return face[np.newaxis, ...]                                # (1, 3, 112, 112)


# ── Embedding extraction ──────────────────────────────────────────────────────
def get_embedding(session: ort.InferenceSession, face: np.ndarray) -> np.ndarray:
    """Run ArcFace inference, return L2-normalised embedding."""
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: face})[0][0]  # (512,)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
    return embedding


# ── LFW pair parsing ──────────────────────────────────────────────────────────
def parse_pairs(pairs_file: Path) -> list[tuple[str, str, int]]:
    """
    Parse LFW pairs.txt into list of (img_path_1, img_path_2, label).
    label=1 for genuine, 0 for impostor.
    """
    pairs = []
    with open(pairs_file) as f:
        lines = f.readlines()

    n_folds, n_per_fold = map(int, lines[0].strip().split())
    idx = 1

    for _ in range(n_folds):
        # Genuine pairs
        for _ in range(n_per_fold):
            parts = lines[idx].strip().split()
            name, n1, n2 = parts[0], int(parts[1]), int(parts[2])
            p1 = LFW_DIR / name / f"{name}_{n1:04d}.jpg"
            p2 = LFW_DIR / name / f"{name}_{n2:04d}.jpg"
            pairs.append((str(p1), str(p2), 1))
            idx += 1
        # Impostor pairs
        for _ in range(n_per_fold):
            parts = lines[idx].strip().split()
            n1, n2 = int(parts[1]), int(parts[3])
            name1, name2 = parts[0], parts[2]
            p1 = LFW_DIR / name1 / f"{name1}_{n1:04d}.jpg"
            p2 = LFW_DIR / name2 / f"{name2}_{n2:04d}.jpg"
            pairs.append((str(p1), str(p2), 0))
            idx += 1

    log.info(f"Parsed {len(pairs)} pairs ({n_folds} folds, {n_per_fold} per class per fold)")
    return pairs


# ── Demographic attribute loading ─────────────────────────────────────────────
def load_attributes(attr_file: Path) -> dict[str, dict]:
    """
    Load LFW Attributes annotations.
    Returns dict mapping image filename stem to attribute dict.
    Attributes include: Male, Asian, White, Black, etc.
    """
    attributes = {}
    if not attr_file.exists():
        log.warning(f"Attribute file not found at {attr_file}. Skipping demographic analysis.")
        return attributes

    with open(attr_file) as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)                    # first row: "# LFW Attribute Values..." comment — skip
        header = next(reader)           # second row: column names (leading "#", person, imagenum)
        attr_names = header[3:]         # skip #, person, imagenum columns

        for row in reader:
            if len(row) < 3:
                continue
            person    = row[0].strip().replace(" ", "_")
            image_num = int(row[1].strip())
            img_key   = f"{person}_{image_num:04d}"
            scores    = {attr_names[i]: float(row[i + 2]) for i in range(len(attr_names))}
            attributes[img_key] = scores

    log.info(f"Loaded attributes for {len(attributes)} images")
    return attributes


def get_subgroup(attributes: dict, img_path: str) -> str:
    """
    Map an image to a demographic subgroup label using LFW Attributes.
    Uses perceived race attributes: Asian, Black, White (positive score = present).
    Falls back to 'Unknown' if not in attribute file.
    """
    stem = Path(img_path).stem          # e.g. "George_W_Bush_0001"
    if stem not in attributes:
        return "Unknown"

    scores = attributes[stem]
    race_attrs = {
        "Asian":  scores.get("Asian", 0),
        "Black":  scores.get("Black", 0),
        "White":  scores.get("White", 0),
    }
    return max(race_attrs, key=race_attrs.get)


# ── Embedding cache ───────────────────────────────────────────────────────────
def build_embedding_cache(
    session: ort.InferenceSession,
    image_paths: list[str],
) -> dict[str, np.ndarray | None]:
    """Pre-compute embeddings for all unique images."""
    unique_paths = list(set(image_paths))
    cache = {}
    failed = 0

    for path in tqdm(unique_paths, desc="Computing embeddings"):
        img = cv2.imread(path)
        if img is None:
            log.warning(f"Could not read image: {path}")
            cache[path] = None
            failed += 1
            continue
        face = preprocess(img)
        if face is None:
            cache[path] = None
            failed += 1
            continue
        try:
            cache[path] = get_embedding(session, face)
        except Exception as e:
            log.warning(f"Embedding error for {path}: {e}")
            cache[path] = None
            failed += 1

    log.info(f"Embedding cache built: {len(cache) - failed}/{len(cache)} successful")
    return cache


# ── Cosine similarity ─────────────────────────────────────────────────────────
def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised embeddings."""
    return float(np.dot(e1, e2))


# ── LFW 10-fold evaluation ────────────────────────────────────────────────────
def lfw_10fold(
    pairs: list[tuple[str, str, int]],
    cache: dict[str, np.ndarray | None],
) -> dict:
    """
    Standard LFW restricted 10-fold cross-validation.
    Threshold is tuned on 9 folds, evaluated on the held-out fold.
    Returns accuracy, std, and all scores/labels.
    """
    n_folds = 10
    fold_size = len(pairs) // n_folds
    accs = []
    all_scores, all_labels = [], []

    for fold in range(n_folds):
        val_pairs  = pairs[fold * fold_size: (fold + 1) * fold_size]
        train_pairs = pairs[:fold * fold_size] + pairs[(fold + 1) * fold_size:]

        # Compute scores
        def score_pairs(pair_list):
            scores, labels, valid = [], [], []
            for p1, p2, label in pair_list:
                e1, e2 = cache.get(p1), cache.get(p2)
                if e1 is None or e2 is None:
                    continue
                scores.append(cosine_similarity(e1, e2))
                labels.append(label)
                valid.append((p1, p2, label))
            return np.array(scores), np.array(labels), valid

        train_scores, train_labels, _ = score_pairs(train_pairs)
        val_scores,   val_labels,   _ = score_pairs(val_pairs)

        # Find best threshold on training folds
        thresholds = np.linspace(-1, 1, 1000)
        best_acc, best_thresh = 0, 0
        for t in thresholds:
            preds = (train_scores >= t).astype(int)
            acc = np.mean(preds == train_labels)
            if acc > best_acc:
                best_acc, best_thresh = acc, t

        # Evaluate on held-out fold
        val_preds = (val_scores >= best_thresh).astype(int)
        fold_acc = np.mean(val_preds == val_labels)
        accs.append(fold_acc)
        all_scores.extend(val_scores.tolist())
        all_labels.extend(val_labels.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std":  float(np.std(accs)),
        "fold_accs":     accs,
        "all_scores":    all_scores,
        "all_labels":    all_labels,
    }


# ── Fairness metrics ──────────────────────────────────────────────────────────
def compute_fairness_metrics(
    pairs: list[tuple[str, str, int]],
    cache: dict[str, np.ndarray | None],
    attributes: dict[str, dict],
    threshold: float,
) -> dict:
    """
    Compute per-subgroup FMR and FNMR at a given threshold.
    FMR  = false match rate    (impostor pairs accepted)
    FNMR = false non-match rate (genuine pairs rejected)
    """
    subgroup_data: dict[str, dict] = {}

    for p1, p2, label in pairs:
        e1, e2 = cache.get(p1), cache.get(p2)
        if e1 is None or e2 is None:
            continue

        score = cosine_similarity(e1, e2)
        pred  = int(score >= threshold)

        # Assign subgroup based on first image in pair
        sg = get_subgroup(attributes, p1)
        if sg not in subgroup_data:
            subgroup_data[sg] = {"impostor_scores": [], "genuine_scores": []}

        if label == 0:
            subgroup_data[sg]["impostor_scores"].append(score)
        else:
            subgroup_data[sg]["genuine_scores"].append(score)

    results = {}
    for sg, data in subgroup_data.items():
        imp = np.array(data["impostor_scores"])
        gen = np.array(data["genuine_scores"])

        fmr  = float(np.mean(imp >= threshold)) if len(imp) > 0 else None
        fnmr = float(np.mean(gen <  threshold)) if len(gen) > 0 else None

        results[sg] = {
            "FMR":            fmr,
            "FNMR":           fnmr,
            "n_impostor":     len(imp),
            "n_genuine":      len(gen),
        }

    # Fairness Discrepancy Rate (gap between best and worst subgroup)
    fmr_vals  = [v["FMR"]  for v in results.values() if v["FMR"]  is not None]
    fnmr_vals = [v["FNMR"] for v in results.values() if v["FNMR"] is not None]

    results["FDR_FMR"]  = float(max(fmr_vals)  - min(fmr_vals))  if fmr_vals  else None
    results["FDR_FNMR"] = float(max(fnmr_vals) - min(fnmr_vals)) if fnmr_vals else None

    return results


# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, threshold: float, n: int = 1000) -> dict:
    """Bootstrap 95% CI for accuracy at a fixed threshold."""
    accs = []
    rng = np.random.default_rng(42)
    for _ in range(n):
        idx  = rng.integers(0, len(scores), len(scores))
        preds = (scores[idx] >= threshold).astype(int)
        accs.append(np.mean(preds == labels[idx]))
    return {"ci_lower": float(np.percentile(accs, 2.5)), "ci_upper": float(np.percentile(accs, 97.5))}


# ── Latency measurement ───────────────────────────────────────────────────────
def measure_latency(session: ort.InferenceSession, n_warmup: int = 100, n_runs: int = 1000) -> dict:
    """Measure end-to-end embedding inference latency (ms)."""
    import time
    dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
    input_name = session.get_inputs()[0].name

    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "latency_median_ms": float(np.median(times)),
        "latency_p95_ms":    float(np.percentile(times, 95)),
        "latency_mean_ms":   float(np.mean(times)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== Baseline Pipeline: FP32 ResNet-50 + ArcFace ===")

    # 1. Load model
    session = load_arcface(WEIGHTS_PATH)

    # 2. Parse LFW pairs
    pairs = parse_pairs(PAIRS_FILE)

    # 3. Build embedding cache
    all_paths = [p for pair in pairs for p in pair[:2]]
    cache = build_embedding_cache(session, all_paths)

    # 4. LFW 10-fold evaluation
    log.info("Running 10-fold evaluation...")
    results = lfw_10fold(pairs, cache)
    log.info(f"Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

    # 5. Set operating threshold (EER point on full set)
    fpr, tpr, thresholds = roc_curve(results["all_labels"], results["all_scores"])
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_threshold = float(thresholds[eer_idx])
    log.info(f"EER threshold: {eer_threshold:.4f}")

    # 6. Bootstrap CI
    ci = bootstrap_ci(results["all_scores"], results["all_labels"], eer_threshold)
    log.info(f"95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # 7. Fairness metrics
    attributes = load_attributes(ATTR_FILE)
    if attributes:
        log.info("Computing fairness metrics...")
        fairness = compute_fairness_metrics(pairs, cache, attributes, eer_threshold)
        log.info("Fairness results:")
        for k, v in fairness.items():
            log.info(f"  {k}: {v}")
    else:
        fairness = {}

    # 8. Latency
    log.info("Measuring latency...")
    latency = measure_latency(session)
    log.info(f"Latency: {latency['latency_median_ms']:.2f} ms (median), {latency['latency_p95_ms']:.2f} ms (p95)")

    # 9. Save results
    output = {
        "model":          "ResNet-50 ArcFace FP32 (baseline)",
        "accuracy_mean":  results["accuracy_mean"],
        "accuracy_std":   results["accuracy_std"],
        "eer_threshold":  eer_threshold,
        "bootstrap_ci":   ci,
        "fairness":       fairness,
        "latency":        latency,
    }

    import json
    out_path = RESULTS_DIR / "baseline_resnet50_fp32.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()