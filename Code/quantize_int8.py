"""
Post-training static INT8 quantisation of the ArcFace ResNet-50 backbone.
==========================================================================
Uses ONNX Runtime's static quantization (QOperator, QDQ-style calibration)
with a real calibration set drawn from LFW: images are detected + aligned
with the SAME SCRFD detector and alignment used by the eval pipeline
(imported unchanged from baseline_pipeline.py), so the calibration
distribution matches what the model sees at eval time.

Static (not dynamic) quantization is used because the backbone is
Conv-dominated (53 Conv vs 1 Gemm) — dynamic quantization mainly helps
MatMul/Gemm-heavy models and would leave almost all of the compute in FP32.

Output: weights_variants/w600k_r50_int8.onnx
"""

import sys
import random
import logging
from pathlib import Path

import numpy as np
import cv2
import onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SRC_WEIGHTS = Path("~/.insightface/models/buffalo_l/w600k_r50.onnx").expanduser()
OUT_DIR     = Path("weights_variants")
OUT_WEIGHTS = OUT_DIR / "w600k_r50_int8.onnx"
N_CALIBRATION_IMAGES = 200
RANDOM_SEED = 42


def build_calibration_image_list(bp) -> list[str]:
    """Reuse baseline_pipeline's pair parsing to get a representative,
    unmodified sample of LFW image paths for calibration."""
    if not bp.PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"Calibration data unavailable: pairs file not found at {bp.PAIRS_FILE}. "
            "Falling back to dynamic quantization is recommended if this cannot be resolved."
        )
    pairs = bp.parse_pairs(bp.PAIRS_FILE)
    all_paths = sorted({p for pair in pairs for p in pair[:2]})
    missing = [p for p in all_paths if not Path(p).exists()]
    if len(missing) == len(all_paths):
        raise FileNotFoundError(
            f"None of the {len(all_paths)} LFW pair images exist under {bp.LFW_DIR}. "
            "Check that the LFW dataset has been extracted."
        )
    random.Random(RANDOM_SEED).shuffle(all_paths)
    sample = [p for p in all_paths if Path(p).exists()][:N_CALIBRATION_IMAGES]
    if len(sample) < 20:
        raise RuntimeError(
            f"Only {len(sample)} usable calibration images found (need >= 20 for a "
            "meaningful calibration set)."
        )
    log.info(f"Calibration set: {len(sample)} images (requested {N_CALIBRATION_IMAGES})")
    return sample


def make_calibration_reader(bp, image_paths: list[str], input_name: str):
    from onnxruntime.quantization import CalibrationDataReader

    class LFWCalibrationReader(CalibrationDataReader):
        def __init__(self):
            faces = []
            failed = 0
            for p in image_paths:
                img = cv2.imread(p)
                if img is None:
                    failed += 1
                    continue
                face = bp.preprocess(img)  # unchanged eval preprocessing: SCRFD detect + align + normalise
                if face is None:
                    failed += 1
                    continue
                faces.append(face)
            if not faces:
                raise RuntimeError("No calibration images could be preprocessed (0 faces detected).")
            log.info(f"Preprocessed {len(faces)}/{len(image_paths)} calibration images "
                      f"({failed} failed detection/read)")
            self._iter = iter({input_name: f} for f in faces)

        def get_next(self):
            return next(self._iter, None)

    return LFWCalibrationReader()


def upgrade_opset(src_path: Path, target_opset: int = 13) -> Path:
    """
    Per-channel QuantizeLinear/DequantizeLinear (needed for accurate CNN
    quantization) requires opset >= 13; the source model is opset 11.
    Convert it once and quantize from the upgraded copy.
    """
    from onnx import version_converter

    model = onnx.load(str(src_path))
    current = model.opset_import[0].version
    if current >= target_opset:
        return src_path

    log.info(f"Upgrading model opset {current} -> {target_opset} for per-channel quantization")
    upgraded = version_converter.convert_version(model, target_opset)
    onnx.checker.check_model(upgraded)
    upgraded_path = OUT_DIR / "w600k_r50_opset13_fp32.onnx"
    OUT_DIR.mkdir(exist_ok=True)
    onnx.save(upgraded, str(upgraded_path))
    return upgraded_path


def quantize(src_path: Path, out_path: Path):
    from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod

    if not src_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found at {src_path}")

    import baseline_pipeline as bp  # reuse eval-identical preprocessing, no eval-logic changes

    upgraded_path = upgrade_opset(src_path)

    session_input_name = onnx.load(str(upgraded_path)).graph.input[0].name
    image_paths = build_calibration_image_list(bp)
    reader = make_calibration_reader(bp, image_paths, session_input_name)

    OUT_DIR.mkdir(exist_ok=True)
    log.info("Running static INT8 quantization (this can take a few minutes)...")
    quantize_static(
        model_input=str(upgraded_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
    )
    log.info(f"Saved INT8 model to {out_path}")


def sanity_check_inference(out_path: Path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
    out = sess.run(None, {input_name: dummy})[0]
    if out.shape != (1, 512):
        raise RuntimeError(f"INT8 model output shape {out.shape} != expected (1, 512)")
    log.info(f"Sanity check OK — output shape {out.shape}")


def main():
    try:
        quantize(SRC_WEIGHTS, OUT_WEIGHTS)
    except Exception as e:
        log.error(f"INT8 quantization failed: {e}")
        sys.exit(1)

    try:
        sanity_check_inference(OUT_WEIGHTS)
    except Exception as e:
        log.error(f"INT8 model failed sanity-check inference: {e}")
        sys.exit(1)

    orig_size = SRC_WEIGHTS.stat().st_size / 1e6
    new_size = OUT_WEIGHTS.stat().st_size / 1e6
    log.info(f"Model size: {orig_size:.1f} MB -> {new_size:.1f} MB "
              f"({(1 - new_size / orig_size):.1%} smaller)")


if __name__ == "__main__":
    main()
