"""
Local VLM Service — image captioning and understanding.

Architecture decisions:
  1. Primary model: SmolVLM-500M-Instruct — fast, 500MB, good for
     batch captioning during PDF indexing (~1-2s/image on CPU).
  2. GPT-4o-mini fallback for hard cases (confidence < threshold).
     Cost: ~$0.001/image, expected usage <5% of images.
  3. Lazy loading — VLM only loads when first needed, not at startup.
     This keeps startup fast and RAM usage low until images are processed.
  4. Automatic unloading after idle period to free RAM.
  5. Caption caching — SHA256(image_bytes) → caption stored in Redis
     so we never caption the same image twice.
"""

from __future__ import annotations

import hashlib
import io
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


class LocalVLM:
    """
    Vision-Language Model for generating image captions/descriptions.
    Lazy-loaded to conserve RAM. Thread-safe.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._model_name = cfg.vlm_model_name
        self._max_new_tokens = cfg.vlm_max_new_tokens
        self._confidence_threshold = cfg.vlm_confidence_threshold

        # Model loaded lazily
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._idle_timeout = 300  # Unload after 5 min idle
        self._ready = False

    def _ensure_loaded(self) -> None:
        """Load model if not already in memory."""
        if self._model is not None:
            self._last_used = time.monotonic()
            return

        with self._lock:
            if self._model is not None:
                self._last_used = time.monotonic()
                return

            _log.info("vlm_loading", model=self._model_name)
            with timed("vlm_model_load") as t:
                import torch
                from transformers import AutoModelForVision2Seq, AutoProcessor

                self._processor = AutoProcessor.from_pretrained(
                    self._model_name,
                    trust_remote_code=True,
                )
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self._model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                self._model.eval()

            self._ready = True
            self._last_used = time.monotonic()
            _log.info("vlm_loaded", model=self._model_name, load_ms=round(t["ms"], 2))

    def unload(self) -> None:
        """Free model memory. Called when idle too long."""
        with self._lock:
            if self._model is not None:
                del self._model
                del self._processor
                self._model = None
                self._processor = None
                self._ready = False
                _log.info("vlm_unloaded")

                # Force garbage collection
                import gc
                gc.collect()

    def maybe_unload(self) -> None:
        """Unload if idle for too long. Called periodically."""
        if self._model is not None and self._last_used > 0:
            idle = time.monotonic() - self._last_used
            if idle > self._idle_timeout:
                self.unload()

    # ── Captioning ──────────────────────────────────────────

    def caption_image(
        self,
        image: Image.Image,
        prompt: str = "Describe this image in detail for search indexing. Include all visible text, objects, diagrams, charts, and their relationships.",
    ) -> Tuple[str, float]:
        """
        Generate a caption for a single image.

        Returns:
            (caption_text, confidence_score) tuple.
            Confidence is based on average token log probability.
        """
        self._ensure_loaded()

        with timed("vlm_caption") as t:
            import torch

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text_prompt = self._processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self._processor(
                text=text_prompt,
                images=[image],
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )

            # Decode only the generated tokens (skip prompt tokens)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            caption = self._processor.decode(generated_ids, skip_special_tokens=True)

            # Simple confidence heuristic: longer, more detailed = higher confidence
            confidence = min(1.0, len(caption.split()) / 20.0)

        metrics.record("vlm_caption_ms", t["ms"])
        self._last_used = time.monotonic()

        return caption.strip(), confidence

    def caption_batch(
        self,
        images: List[Image.Image],
        prompt: str = "Describe this image in detail for search indexing.",
    ) -> List[Tuple[str, float]]:
        """
        Caption a batch of images sequentially.
        (True batching requires padding which hurts quality for variable-size images.)

        Returns:
            List of (caption, confidence) tuples.
        """
        results = []
        for i, img in enumerate(images):
            try:
                caption, conf = self.caption_image(img, prompt)
                results.append((caption, conf))
                if (i + 1) % 10 == 0:
                    _log.info("vlm_caption_progress", done=i + 1, total=len(images))
            except Exception as e:
                _log.warning("vlm_caption_failed", index=i, error=str(e))
                results.append(("", 0.0))

        return results

    def describe_image(
        self,
        image: Image.Image,
        question: str,
    ) -> str:
        """
        Answer a question about an image (query-time VLM).

        Args:
            image: PIL Image to analyze.
            question: User's question about the image.

        Returns:
            Answer string.
        """
        self._ensure_loaded()

        with timed("vlm_describe") as t:
            import torch

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text_prompt = self._processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self._processor(
                text=text_prompt,
                images=[image],
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            answer = self._processor.decode(generated_ids, skip_special_tokens=True)

        metrics.record("vlm_describe_ms", t["ms"])
        self._last_used = time.monotonic()

        return answer.strip()

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# ── GPT-4o Fallback ─────────────────────────────────────────

async def gpt4o_caption(
    image: Image.Image,
    prompt: str = "Describe this image in detail for search indexing. Include all visible text, objects, diagrams, and their relationships.",
) -> str:
    """
    Fallback to GPT-4o-mini for hard images.
    Cost: ~$0.001 per image.
    """
    import base64
    import openai

    cfg = get_settings()
    if not cfg.openai_api_key:
        _log.warning("gpt4o_fallback_no_key")
        return ""

    # Convert PIL to base64 JPEG
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = openai.AsyncOpenAI(api_key=cfg.openai_api_key)

    with timed("gpt4o_caption") as t:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=256,
        )

    caption = response.choices[0].message.content or ""
    metrics.record("gpt4o_caption_ms", t["ms"])

    _log.info("gpt4o_caption_complete", length=len(caption), ms=round(t["ms"], 2))
    return caption.strip()


# ── Image Hash Utility ──────────────────────────────────────

def image_hash(image: Image.Image) -> str:
    """SHA256 hash of image pixels for cache key."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()[:32]


# ── Caption Cache ───────────────────────────────────────────

class CaptionCache:
    """
    Redis-backed cache for VLM captions.
    Key: sha256(image_bytes)[:32], Value: caption string.
    """

    def __init__(self) -> None:
        self._prefix = "vlm_caption:"
        self._redis = None

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            import redis
            cfg = get_settings()
            self._redis = redis.Redis(
                host=cfg.redis_host,
                port=cfg.redis_port,
                db=cfg.redis_db,
                decode_responses=True,
            )
            self._redis.ping()
            return self._redis
        except Exception:
            return None

    def get(self, img_hash: str) -> Optional[str]:
        """Get cached caption by image hash."""
        r = self._get_redis()
        if r is None:
            return None
        try:
            return r.get(f"{self._prefix}{img_hash}")
        except Exception:
            return None

    def set(self, img_hash: str, caption: str) -> None:
        """Cache a caption. TTL = 30 days."""
        r = self._get_redis()
        if r is None:
            return
        try:
            r.setex(f"{self._prefix}{img_hash}", 2592000, caption)
        except Exception:
            pass


# ── Module-level singleton ──────────────────────────────────

_vlm_instance: Optional[LocalVLM] = None
_cache_instance: Optional[CaptionCache] = None
_vlm_lock = threading.Lock()


def get_vlm() -> LocalVLM:
    """Get or create the singleton LocalVLM."""
    global _vlm_instance
    if _vlm_instance is not None:
        return _vlm_instance
    with _vlm_lock:
        if _vlm_instance is not None:
            return _vlm_instance
        _vlm_instance = LocalVLM()
        return _vlm_instance


def get_caption_cache() -> CaptionCache:
    """Get or create the singleton CaptionCache."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CaptionCache()
    return _cache_instance
