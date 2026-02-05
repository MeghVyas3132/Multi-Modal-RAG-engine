"""
ONNX Model Conversion Script -- Export OpenCLIP text encoder to ONNX format.

Architecture decisions:
  1. We export ONLY the text encoder, not the image encoder. The text
     encoder is the hot path (every search request). Image encoding is
     offline-only and can stay in PyTorch.
  2. The tokenizer remains in Python (open_clip.get_tokenizer). It's
     not a neural network -- it's a fast vocabulary lookup. No benefit
     from ONNX export.
  3. We export with opset 17 for maximum operator support and run
     optimization passes (constant folding, operator fusion) to reduce
     inference overhead.
  4. Both FP32 and FP16 models are exported. FP16 is ~2x smaller and
     ~1.5x faster on CPUs with AVX512, with negligible accuracy loss
     for normalized embeddings.
  5. Validation: after export, we run 10 random queries through both
     PyTorch and ONNX and verify outputs match within epsilon=0.001.

Usage:
  python -m scripts.convert_to_onnx
  python -m scripts.convert_to_onnx --model ViT-H-14 --output-dir models/onnx
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import open_clip

# ONNX imports -- these are optional dependencies
try:
    import onnx
    from onnxruntime.transformers.optimizer import optimize_model
    import onnxruntime as ort
except ImportError:
    print("ERROR: Install onnx and onnxruntime packages first.")
    print("  pip install onnx onnxruntime onnxruntime-extensions")
    sys.exit(1)


def load_clip_model(
    model_name: str,
    pretrained: str,
) -> Tuple[torch.nn.Module, object, int]:
    """
    Load the OpenCLIP model and return the text encoder component.

    Returns:
        (model, tokenizer, vector_dim)
    """
    print(f"Loading OpenCLIP model: {model_name} / {pretrained}")

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device="cpu",  # Export from CPU to avoid CUDA dependencies in ONNX
    )
    model.eval()

    tokenizer = open_clip.get_tokenizer(model_name)

    # Determine vector dimension from a dummy forward pass
    dummy_tokens = tokenizer(["test"]).to("cpu")
    with torch.no_grad():
        dummy_out = model.encode_text(dummy_tokens)
    vector_dim = dummy_out.shape[-1]

    print(f"  Model loaded. Vector dimension: {vector_dim}")
    return model, tokenizer, vector_dim


class TextEncoderWrapper(torch.nn.Module):
    """
    Wrapper that isolates the text encoder for ONNX export.

    The full CLIP model contains both image and text towers. We only
    need the text tower for search queries. This wrapper exposes a
    clean forward() that takes token IDs and returns L2-normalized
    text embeddings.
    """

    def __init__(self, clip_model: torch.nn.Module) -> None:
        super().__init__()
        self.clip_model = clip_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs from the OpenCLIP tokenizer.
                       Shape: (batch_size, context_length)

        Returns:
            L2-normalized text embeddings.
            Shape: (batch_size, vector_dim)
        """
        # Run through the text encoder
        text_features = self.clip_model.encode_text(input_ids)
        # L2 normalize so cosine similarity = dot product in Qdrant
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features


def export_onnx(
    model: torch.nn.Module,
    tokenizer: object,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """
    Export the text encoder to ONNX format.

    The model is exported with a fixed context length (77 tokens,
    which is the default for CLIP). Batch size is dynamic.
    """
    wrapper = TextEncoderWrapper(model)
    wrapper.eval()

    # Create dummy input -- CLIP uses fixed context length of 77 tokens
    dummy_input = tokenizer(["a photo of a cat"]).to("cpu")
    context_length = dummy_input.shape[1]  # Should be 77

    print(f"  Exporting to ONNX (opset {opset_version}, context_length={context_length})")

    # Export with dynamic batch size but fixed sequence length
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        output_path,
        export_params=True,          # Store trained weights in the ONNX file
        opset_version=opset_version, # ONNX opset version
        do_constant_folding=True,    # Fold constants for faster inference
        input_names=["input_ids"],   # Name for the input tensor
        output_names=["embeddings"], # Name for the output tensor
        dynamic_axes={
            # Allow variable batch size (dim 0) for batch inference
            "input_ids": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
    )

    # Verify the exported model is valid ONNX
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Exported: {output_path} ({file_size_mb:.1f} MB)")


def optimize_onnx_model(input_path: str, output_path: str) -> None:
    """
    Run ONNX Runtime optimization passes on the exported model.

    Optimizations include:
      - Constant folding (pre-compute static expressions)
      - Operator fusion (merge sequential ops into fused kernels)
      - Redundant node elimination
    """
    print(f"  Running optimization passes...")

    optimized = optimize_model(
        input_path,
        model_type="clip",
        num_heads=0,       # Auto-detect from model
        hidden_size=0,     # Auto-detect from model
        opt_level=2,       # Level 2: extended optimizations
    )
    optimized.save_model_to_file(output_path)

    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    opt_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Optimized: {output_path} ({orig_size:.1f} MB -> {opt_size:.1f} MB)")


def convert_to_fp16(input_path: str, output_path: str) -> None:
    """
    Convert FP32 ONNX model to FP16 for reduced memory and faster inference.

    FP16 is ~2x smaller and ~1.5x faster on modern CPUs with FP16 support.
    Accuracy loss is negligible for normalized embeddings (<0.001 cosine delta).
    """
    from onnxruntime.transformers.float16 import convert_float_to_float16

    print(f"  Converting to FP16...")

    model = onnx.load(input_path)
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,  # Keep inputs/outputs as FP32 for compatibility
    )
    onnx.save(model_fp16, output_path)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  FP16 model: {output_path} ({file_size_mb:.1f} MB)")


def validate_onnx(
    pytorch_model: torch.nn.Module,
    tokenizer: object,
    onnx_path: str,
    num_samples: int = 10,
    epsilon: float = 0.001,
) -> bool:
    """
    Validate ONNX model outputs match PyTorch within epsilon.

    Runs num_samples queries through both backends and compares
    the output vectors using max absolute difference.

    Returns True if all samples pass.
    """
    print(f"\n  Validating ONNX model against PyTorch ({num_samples} samples)...")

    wrapper = TextEncoderWrapper(pytorch_model)
    wrapper.eval()

    # Create ONNX Runtime session
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    test_queries = [
        "a photo of a cat",
        "sunset over the ocean",
        "a person riding a bicycle",
        "abstract painting with bright colors",
        "a red sports car on a highway",
        "mountain landscape with snow",
        "a dog playing fetch in the park",
        "city skyline at night",
        "fresh fruit on a wooden table",
        "astronaut floating in space",
    ][:num_samples]

    all_passed = True
    max_diff_seen = 0.0

    for query in test_queries:
        tokens = tokenizer([query]).to("cpu")

        # PyTorch inference
        with torch.no_grad():
            pt_output = wrapper(tokens).numpy()

        # ONNX inference
        ort_output = session.run(
            None,
            {"input_ids": tokens.numpy().astype(np.int64)},
        )[0]

        # Compare outputs
        max_diff = np.max(np.abs(pt_output - ort_output))
        max_diff_seen = max(max_diff_seen, max_diff)

        if max_diff > epsilon:
            print(f"    FAIL: '{query}' max_diff={max_diff:.6f} > epsilon={epsilon}")
            all_passed = False
        else:
            print(f"    PASS: '{query}' max_diff={max_diff:.6f}")

    print(f"  Max difference across all samples: {max_diff_seen:.6f}")
    print(f"  Validation: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def benchmark_comparison(
    pytorch_model: torch.nn.Module,
    tokenizer: object,
    onnx_path: str,
    num_iterations: int = 100,
) -> None:
    """Quick latency comparison between PyTorch and ONNX."""
    print(f"\n  Benchmarking ({num_iterations} iterations)...")

    wrapper = TextEncoderWrapper(pytorch_model)
    wrapper.eval()

    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    query = "a photo of a cat sitting on a couch"
    tokens = tokenizer([query]).to("cpu")

    # Warmup both backends
    for _ in range(5):
        with torch.no_grad():
            wrapper(tokens)
        session.run(None, {"input_ids": tokens.numpy().astype(np.int64)})

    # Benchmark PyTorch
    pt_times = []
    for _ in range(num_iterations):
        start = time.perf_counter_ns()
        with torch.no_grad():
            wrapper(tokens)
        pt_times.append((time.perf_counter_ns() - start) / 1_000_000)

    # Benchmark ONNX
    ort_times = []
    for _ in range(num_iterations):
        start = time.perf_counter_ns()
        session.run(None, {"input_ids": tokens.numpy().astype(np.int64)})
        ort_times.append((time.perf_counter_ns() - start) / 1_000_000)

    pt_times.sort()
    ort_times.sort()

    def p(arr, pct):
        return round(arr[int(len(arr) * pct / 100)], 2)

    print(f"\n  {'Metric':<15} {'PyTorch (ms)':>15} {'ONNX (ms)':>15} {'Speedup':>10}")
    print(f"  {'-'*55}")
    print(f"  {'P50':<15} {p(pt_times, 50):>15} {p(ort_times, 50):>15} {p(pt_times, 50)/max(p(ort_times, 50), 0.01):>9.1f}x")
    print(f"  {'P95':<15} {p(pt_times, 95):>15} {p(ort_times, 95):>15} {p(pt_times, 95)/max(p(ort_times, 95), 0.01):>9.1f}x")
    print(f"  {'P99':<15} {p(pt_times, 99):>15} {p(ort_times, 99):>15} {p(pt_times, 99)/max(p(ort_times, 99), 0.01):>9.1f}x")
    print(f"  {'Mean':<15} {round(sum(pt_times)/len(pt_times), 2):>15} {round(sum(ort_times)/len(ort_times), 2):>15}")


def main():
    parser = argparse.ArgumentParser(description="Export OpenCLIP text encoder to ONNX")
    parser.add_argument("--model", default="ViT-H-14", help="OpenCLIP model name")
    parser.add_argument("--pretrained", default="laion2b_s32b_b79k", help="Pretrained weights tag")
    parser.add_argument("--output-dir", default="models/onnx", help="Output directory for ONNX files")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--skip-fp16", action="store_true", help="Skip FP16 conversion")
    parser.add_argument("--skip-validation", action="store_true", help="Skip output validation")
    parser.add_argument("--benchmark-iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive file names from model name
    safe_name = args.model.lower().replace("-", "_").replace("/", "_")
    raw_path = str(output_dir / f"clip_{safe_name}_text_raw.onnx")
    fp32_path = str(output_dir / f"clip_{safe_name}_text_fp32.onnx")
    fp16_path = str(output_dir / f"clip_{safe_name}_text_fp16.onnx")

    print("=" * 60)
    print("OpenCLIP to ONNX Conversion")
    print("=" * 60)

    # Step 1: Load model
    model, tokenizer, vector_dim = load_clip_model(args.model, args.pretrained)

    # Step 2: Export raw ONNX
    print(f"\nStep 2: Export to ONNX")
    export_onnx(model, tokenizer, raw_path, opset_version=args.opset)

    # Step 3: Optimize
    print(f"\nStep 3: Optimize ONNX model")
    try:
        optimize_onnx_model(raw_path, fp32_path)
    except Exception as e:
        print(f"  Optimization failed ({e}), using raw model as FP32")
        import shutil
        shutil.copy(raw_path, fp32_path)

    # Step 4: Convert to FP16
    if not args.skip_fp16:
        print(f"\nStep 4: Convert to FP16")
        try:
            convert_to_fp16(fp32_path, fp16_path)
        except Exception as e:
            print(f"  FP16 conversion failed: {e}")

    # Step 5: Validate
    if not args.skip_validation:
        print(f"\nStep 5: Validate ONNX output")
        validate_onnx(model, tokenizer, fp32_path)

    # Step 6: Benchmark
    print(f"\nStep 6: Benchmark comparison")
    benchmark_comparison(model, tokenizer, fp32_path, num_iterations=args.benchmark_iters)

    # Cleanup raw (unoptimized) file
    if os.path.exists(raw_path) and os.path.exists(fp32_path) and raw_path != fp32_path:
        os.remove(raw_path)
        print(f"\n  Cleaned up: {raw_path}")

    print(f"\n{'=' * 60}")
    print(f"Conversion complete. Models saved to: {output_dir}/")
    print(f"  FP32: {fp32_path}")
    if not args.skip_fp16 and os.path.exists(fp16_path):
        print(f"  FP16: {fp16_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
