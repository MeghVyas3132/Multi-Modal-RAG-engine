"""
Embedding Service Factory -- runtime backend selection.

The factory inspects settings.use_onnx to determine which embedding
backend to instantiate. Both backends expose the same interface:
  - encode_text(text: str) -> np.ndarray
  - encode_text_batch(texts: List[str]) -> np.ndarray
  - is_ready: bool
  - vector_dim: int
  - device: str (torch device or ONNX provider)

Fallback logic:
  1. If use_onnx=True and ONNX model exists and onnxruntime is installed
     -> use ONNXCLIPEmbedder
  2. Otherwise -> fall back to CLIPEmbedder (PyTorch)
"""

from __future__ import annotations

import os
from typing import Union

from configs.settings import get_settings
from utils.logger import get_logger

_log = get_logger(__name__)

# Type alias for either embedder backend
EmbedderType = Union["CLIPEmbedder", "ONNXCLIPEmbedder"]  # noqa: F821


def create_embedder_auto() -> EmbedderType:
    """
    Factory function: create the appropriate embedder based on settings.

    Decision tree:
      1. Check if USE_ONNX is True
      2. Check if onnxruntime is importable
      3. Check if the ONNX model file exists on disk
      4. If all pass -> ONNXCLIPEmbedder
      5. Otherwise -> CLIPEmbedder with a warning

    Returns:
        An embedder instance (either ONNX or PyTorch backend).
    """
    cfg = get_settings()

    if cfg.use_onnx:
        _log.info("embedder_factory_attempting_onnx")

        # Check 1: Is onnxruntime installed?
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            _log.warning(
                "embedder_factory_onnx_fallback",
                reason="onnxruntime package not installed",
            )
            return _create_pytorch_embedder()

        # Check 2: Does the ONNX model file exist?
        model_path = cfg.onnx_model_path
        if cfg.use_fp16:
            model_path = model_path.replace("_fp32.onnx", "_fp16.onnx")

        if not os.path.exists(model_path):
            _log.warning(
                "embedder_factory_onnx_fallback",
                reason=f"ONNX model not found at {model_path}",
            )
            return _create_pytorch_embedder()

        # All checks passed -- create ONNX embedder
        try:
            from services.embedding_service.onnx_embedder import create_onnx_embedder
            embedder = create_onnx_embedder()
            _log.info(
                "embedder_factory_onnx_ready",
                provider=embedder.device,
                model=model_path,
            )
            return embedder
        except Exception as e:
            _log.warning(
                "embedder_factory_onnx_failed",
                error=str(e),
                fallback="pytorch",
            )
            return _create_pytorch_embedder()

    # Default: PyTorch backend
    return _create_pytorch_embedder()


def _create_pytorch_embedder() -> "CLIPEmbedder":  # noqa: F821
    """Create the PyTorch-based CLIPEmbedder."""
    from services.embedding_service.embedder import create_embedder
    embedder = create_embedder()
    _log.info(
        "embedder_factory_pytorch_ready",
        device=str(embedder.device),
    )
    return embedder
