"""
ONNX Runtime CLIP Embedding Service -- 2-3x faster CPU inference.

Architecture decisions:
  1. We use ONNX Runtime (ORT) for the text encoder forward pass only.
     The tokenizer stays in Python (open_clip.get_tokenizer) because
     it is a fast vocabulary lookup, not a neural network.
  2. ORT session is configured with execution providers in priority
     order: TensorRT > CUDA > CPU. The first available provider is
     used automatically.
  3. We keep the same public interface as CLIPEmbedder (encode_text,
     encode_text_batch, is_ready, vector_dim, device) so the factory
     can swap them transparently.
  4. Session options are tuned for low-latency single-request inference:
     - intra_op parallelism: 4 threads (within a single operator)
     - inter_op parallelism: 2 threads (between independent operators)
     - execution mode: parallel (run independent ops concurrently)
     - graph optimization: all (constant folding, fusion, etc.)
  5. Warmup: we run 3 dummy inferences at init to trigger ORT's
     internal memory allocation and JIT compilation.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional

import numpy as np

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


class ONNXCLIPEmbedder:
    """
    ONNX Runtime-based CLIP text encoder.

    Drop-in replacement for CLIPEmbedder with 2-3x faster CPU inference.
    Uses the same tokenizer (open_clip) but replaces the PyTorch forward
    pass with an ONNX Runtime session.
    """

    def __init__(self) -> None:
        cfg = get_settings()

        # ── Import dependencies ─────────────────────────────
        # Lazy import so the module can be loaded even if onnxruntime
        # is not installed (factory will fall back to PyTorch).
        try:
            import onnxruntime as ort
            import open_clip
        except ImportError as e:
            raise RuntimeError(
                f"ONNX embedder requires 'onnxruntime' and 'open_clip' packages: {e}"
            )

        # ── Resolve ONNX model path ─────────────────────────
        model_path = cfg.onnx_model_path
        if cfg.use_fp16:
            # Swap FP32 path for FP16 variant
            model_path = model_path.replace("_fp32.onnx", "_fp16.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. "
                f"Run 'python -m scripts.convert_to_onnx' first."
            )

        # ── Configure execution providers ────────────────────
        # Parse comma-separated provider list from config
        requested_providers = [
            p.strip() for p in cfg.onnx_providers.split(",")
        ]

        # Filter to only available providers
        available = ort.get_available_providers()
        providers = [p for p in requested_providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
            _log.warning(
                "onnx_no_requested_providers_available",
                requested=requested_providers,
                available=available,
                fallback="CPUExecutionProvider",
            )

        _log.info(
            "onnx_providers_selected",
            providers=providers,
            available=available,
        )

        # ── Configure session options ────────────────────────
        session_opts = ort.SessionOptions()

        # Thread counts for CPU inference
        session_opts.intra_op_num_threads = cfg.onnx_intra_op_threads
        session_opts.inter_op_num_threads = cfg.onnx_inter_op_threads

        # Execution mode: parallel runs independent ops concurrently
        if cfg.onnx_execution_mode == "parallel":
            session_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            session_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Enable all graph optimizations (constant folding, fusion, etc.)
        session_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Disable memory pattern optimization for more predictable latency
        session_opts.enable_mem_pattern = False

        # ── Create ONNX Runtime session ─────────────────────
        _log.info("onnx_loading_model", path=model_path)
        with timed("onnx_model_load"):
            self._session = ort.InferenceSession(
                model_path,
                sess_options=session_opts,
                providers=providers,
            )

        # ── Introspect input/output names ────────────────────
        # These are set during ONNX export and must match what we feed
        self._input_name = self._session.get_inputs()[0].name      # "input_ids"
        self._output_name = self._session.get_outputs()[0].name    # "embeddings"
        _log.info(
            "onnx_io_names",
            input=self._input_name,
            output=self._output_name,
        )

        # ── Load tokenizer (stays in Python, not in ONNX) ──
        self._tokenizer = open_clip.get_tokenizer(cfg.clip_model_name)

        # ── Store metadata ──────────────────────────────────
        self._vector_dim = cfg.clip_vector_dim
        self._model_path = model_path
        self._active_provider = self._session.get_providers()[0]
        self._ready = True

        # ── Warmup: pre-allocate ORT internal buffers ───────
        self._warmup()

    def _warmup(self, num_passes: int = 3) -> None:
        """
        Run dummy inferences to trigger ORT memory allocation and
        internal JIT compilation. Without this, the first real request
        pays a ~50ms penalty.
        """
        with timed("onnx_warmup"):
            for _ in range(num_passes):
                _ = self.encode_text("warmup query for ONNX runtime")
        _log.info("onnx_warmup_complete", passes=num_passes)

    # ── Public API ──────────────────────────────────────────

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text query into a normalized embedding vector.

        Returns:
            np.ndarray of shape (vector_dim,), dtype float32, L2-normalized.

        This is the hot path. ONNX Runtime replaces PyTorch here for
        2-3x speedup on CPU.
        """
        with timed("onnx_text_encode") as t:
            # Tokenize using OpenCLIP tokenizer (fast Python op, ~0.1ms)
            tokens = self._tokenizer([text]).numpy().astype(np.int64)

            # Run ONNX inference (the hot path -- ~5-15ms CPU vs ~20-50ms PyTorch)
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: tokens},
            )

            # Output is already L2-normalized (done in the ONNX graph)
            vector = outputs[0].squeeze(0).astype(np.float32)

        metrics.record("embedding_latency_ms", t["ms"])
        return vector

    def encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """
        Batch encode multiple text queries.

        Returns:
            np.ndarray of shape (len(texts), vector_dim), dtype float32.
        """
        tokens = self._tokenizer(texts).numpy().astype(np.int64)

        outputs = self._session.run(
            [self._output_name],
            {self._input_name: tokens},
        )

        return outputs[0].astype(np.float32)

    @property
    def vector_dim(self) -> int:
        return self._vector_dim

    @property
    def device(self) -> str:
        """Return the active ONNX execution provider as the 'device'."""
        return self._active_provider

    @property
    def is_ready(self) -> bool:
        return self._ready


# ── Module-level singleton ──────────────────────────────────

_instance: Optional[ONNXCLIPEmbedder] = None
_lock = threading.Lock()


def create_onnx_embedder() -> ONNXCLIPEmbedder:
    """Create or return the singleton ONNXCLIPEmbedder. Thread-safe."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is not None:
            return _instance
        _instance = ONNXCLIPEmbedder()
        return _instance


def get_onnx_embedder() -> ONNXCLIPEmbedder:
    """Return the singleton. Raises if not yet created."""
    if _instance is None:
        raise RuntimeError(
            "ONNXCLIPEmbedder not initialized. Call create_onnx_embedder() at startup."
        )
    return _instance
