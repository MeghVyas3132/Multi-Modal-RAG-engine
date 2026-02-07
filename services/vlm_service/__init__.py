"""
VLM Service Package — Vision-Language Model for image understanding.
"""

from services.vlm_service.local_vlm import LocalVLM, CaptionCache, get_vlm, get_caption_cache

__all__ = [
    "LocalVLM",
    "CaptionCache",
    "get_vlm",
    "get_caption_cache",
]
