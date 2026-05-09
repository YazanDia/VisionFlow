"""
VisionFlow — Task 1: AI Hardware Optimization
==============================================
Converts FastSAM-s.pt → ONNX → TensorRT FP16 engine.
Tuned for RTX 3060 (6 GB VRAM) and TensorRT 10.x.

Usage:
    python convert_to_tensorrt.py --weights FastSAM-s.pt --output fastsam_fp16.engine
"""

import argparse
import os
import sys
import gc
import logging

import torch
import tensorrt as trt
from ultralytics import FastSAM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("VisionFlow.Convert")

# ─── Constants ────────────────────────────────────────────────────────────────
INPUT_H, INPUT_W = 640, 640          # FastSAM-s native resolution
ONNX_OPSET      = 17                 # Required for modern TensorRT
MAX_WORKSPACE   = 4 << 30            # 4 GB scratch; leaves ~2 GB for activations
FP16_ENABLED    = True               # Halves VRAM usage vs FP32


# ─── Step 1: Load model & export to ONNX ──────────────────────────────────────
def export_onnx(weights_path: str, onnx_path: str) -> str:
    """
    Loads FastSAM-s via Ultralytics and exports to ONNX.
    """
    log.info(f"Loading FastSAM-s weights from: {weights_path}")

    try:
        model = FastSAM(weights_path)
    except Exception as exc:
        log.error(f"Failed to load model: {exc}")
        raise

    log.info("Exporting to ONNX (opset %d) …", ONNX_OPSET)
    try:
        # Ultralytics export() returns the saved path
        saved = model.export(
            format="onnx",
            imgsz=[INPUT_H, INPUT_W],
            opset=ONNX_OPSET,
            simplify=True,          # onnx-simplifier folds constants
            dynamic=False,          # Static batch=1 is ~15% faster in TRT
            half=FP16_ENABLED,      # Export weights in FP16
        )
        log.info("ONNX saved to: %s", saved)
    except torch.cuda.OutOfMemoryError:
        log.error("CUDA OOM during ONNX export. Falling back to CPU export.")
        model.to("cpu")
        saved = model.export(
            format="onnx",
            imgsz=[INPUT_H, INPUT_W],
            opset=ONNX_OPSET,
            simplify=True,
            dynamic=False,
            half=False,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Rename to desired output path if needed
    if saved != onnx_path:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        os.rename(saved, onnx_path)
        log.info("Moved ONNX to: %s", onnx_path)

    return onnx_path


# ─── Step 2: Compile ONNX → TensorRT FP16 engine ────────────────────────────
def build_engine(onnx_path: str, engine_path: str) -> None:
    """
    Builds a TensorRT FP16 serialized engine from the ONNX graph.
    Updated for TensorRT 10.x compatibility.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    log.info("Initialising TensorRT builder …")
    
    # Using the builder context managers
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # ── Parser ───────────────────────────────────────────────────────────
    log.info("Parsing ONNX graph …")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("ONNX parse error %d: %s", i, parser.get_error(i))
            raise RuntimeError("ONNX parse failed — see errors above.")

    # ── Config ───────────────────────────────────────────────────────────
    # Set workspace memory pool
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, MAX_WORKSPACE)

    if FP16_ENABLED and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        log.info("FP16 precision enabled (RTX 3060 tensor cores active).")
    else:
        log.warning("FP16 not available on this device; using FP32.")

    # ── Optimization Profile ──────────────────────────────────────────────
    # Required in newer TRT versions to define expected shapes
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    
    # Define (Min, Opt, Max) shapes
    profile.set_shape(
        input_name, 
        (1, 3, INPUT_H, INPUT_W), 
        (1, 3, INPUT_H, INPUT_W), 
        (1, 3, INPUT_H, INPUT_W)
    )
    config.add_optimization_profile(profile)

    # ── Build ─────────────────────────────────────────────────────────────
    log.info("Building engine — this may take 3–10 minutes …")
    try:
        # Build the serialized network directly
        serialized = builder.build_serialized_network(network, config)
    except Exception as trt_err:
        log.error("TensorRT build failed: %s", trt_err)
        raise

    if serialized is None:
        raise RuntimeError(
            "builder.build_serialized_network() returned None. "
            "Check GPU memory and console logs for parsing errors."
        )

    # ── Save ─────────────────────────────────────────────────────────────
    with open(engine_path, "wb") as f:
        f.write(serialized)
    
    log.info("✓ TensorRT engine saved to: %s", engine_path)
    log.info("  Engine size: %.1f MB", os.path.getsize(engine_path) / 1e6)


# ─── Entrypoint ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VisionFlow: FastSAM → TensorRT")
    parser.add_argument("--weights",  default="FastSAM-s.pt",          help="Path to FastSAM-s.pt")
    parser.add_argument("--onnx",     default="FastSAM-s.onnx",        help="Intermediate ONNX path")
    parser.add_argument("--output",   default="fastsam_fp16.engine",   help="Output engine path")
    parser.add_argument("--skip-onnx", action="store_true",            help="Skip ONNX export (use existing)")
    args = parser.parse_args()

    # ── CUDA pre-flight ───────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        log.error("No CUDA device found. TensorRT requires an NVIDIA GPU.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info("CUDA device : %s", device_name)
    log.info("Total VRAM  : %.2f GB", vram_gb)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    if not args.skip_onnx:
        export_onnx(args.weights, args.onnx)

    build_engine(args.onnx, args.output)
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()