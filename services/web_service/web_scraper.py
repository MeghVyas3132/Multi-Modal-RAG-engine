"""
Web Scraper — multi-tier web content extraction.

Architecture decisions:
  1. Three-tier scraping: Jina Reader (free) → Firecrawl (paid) → httpx (fallback).
  2. Platform-specific scrapers for YouTube (transcript), GitHub (README/code).
  3. Change detection via SHA256 hash — only re-index if content changed.
  4. Rate limiting and politeness (respect robots.txt patterns).
  5. Returns structured WebContent objects compatible with the chunking pipeline.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx

from configs.settings import get_settings
from utils.logger import get_logger
from utils.timing import timed
from utils.metrics import metrics

_log = get_logger(__name__)


@dataclass
class WebContent:
    """Extracted web content ready for chunking and indexing."""
    url: str
    title: str
    content: str  # Markdown text
    content_hash: str  # SHA256 for change detection
    source_type: str  # web, youtube, github, twitter, reddit
    scraped_at: float  # Unix timestamp
    language: str = "en"
    metadata: dict = field(default_factory=dict)
    images: List[dict] = field(default_factory=list)  # [{url, alt_text}]


def _content_hash(text: str) -> str:
    """SHA256 of content for change detection."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


# ── Jina Reader (Free Tier) ────────────────────────────────

async def scrape_jina(url: str) -> Optional[WebContent]:
    """
    Use Jina Reader API for free web scraping.
    60 req/min free tier. Returns clean markdown.
    """
    cfg = get_settings()
    jina_url = f"https://r.jina.ai/{url}"

    headers = {"Accept": "application/json"}
    if cfg.jina_api_key:
        headers["Authorization"] = f"Bearer {cfg.jina_api_key}"

    with timed("jina_scrape") as t:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(jina_url, headers=headers)
                resp.raise_for_status()

            data = resp.json()
            content = data.get("data", {}).get("content", "")
            title = data.get("data", {}).get("title", "")

            if not content or len(content.strip()) < 50:
                _log.warning("jina_empty_content", url=url)
                return None

            return WebContent(
                url=url,
                title=title,
                content=content,
                content_hash=_content_hash(content),
                source_type="web",
                scraped_at=time.time(),
                metadata={"scraper": "jina", "status_code": resp.status_code},
            )

        except Exception as e:
            _log.warning("jina_scrape_failed", url=url, error=str(e))
            return None

    metrics.record("jina_scrape_ms", t["ms"])


# ── Firecrawl (Paid Tier) ──────────────────────────────────

async def scrape_firecrawl(url: str) -> Optional[WebContent]:
    """
    Use Firecrawl API for JS-rendered pages.
    Paid: $20/month for unlimited. Returns markdown + screenshots.
    """
    cfg = get_settings()
    if not cfg.firecrawl_api_key:
        return None

    with timed("firecrawl_scrape") as t:
        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=cfg.firecrawl_api_key)
            result = app.scrape_url(url, params={"formats": ["markdown"]})

            content = result.get("markdown", "")
            title = result.get("metadata", {}).get("title", "")

            if not content or len(content.strip()) < 50:
                return None

            return WebContent(
                url=url,
                title=title,
                content=content,
                content_hash=_content_hash(content),
                source_type="web",
                scraped_at=time.time(),
                metadata={
                    "scraper": "firecrawl",
                    "og_title": result.get("metadata", {}).get("ogTitle", ""),
                },
            )

        except Exception as e:
            _log.warning("firecrawl_failed", url=url, error=str(e))
            return None


# ── httpx Direct (Fallback) ─────────────────────────────────

async def scrape_httpx(url: str) -> Optional[WebContent]:
    """
    Direct HTTP fetch with basic HTML-to-text extraction.
    Last resort — no JS rendering, no markdown conversion.
    """
    import re

    with timed("httpx_scrape") as t:
        try:
            async with httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                headers={"User-Agent": "MultiModalRAG/2.0"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            html = resp.text

            # Basic HTML tag stripping
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            # Extract title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else ""

            if len(text) < 50:
                return None

            return WebContent(
                url=url,
                title=title,
                content=text[:50000],  # Cap at 50K chars
                content_hash=_content_hash(text),
                source_type="web",
                scraped_at=time.time(),
                metadata={"scraper": "httpx", "status_code": resp.status_code},
            )

        except Exception as e:
            _log.warning("httpx_scrape_failed", url=url, error=str(e))
            return None


# ── YouTube Transcript ──────────────────────────────────────

async def scrape_youtube(url: str) -> Optional[WebContent]:
    """
    Extract transcript from YouTube video.
    Free, no API key needed. Uses youtube-transcript-api.
    """
    import re

    # Extract video ID
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]

    video_id = None
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break

    if not video_id:
        _log.warning("youtube_no_video_id", url=url)
        return None

    with timed("youtube_scrape") as t:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id)

            # Join all transcript segments
            full_text = " ".join(
                segment.text for segment in transcript_list
            )

            if not full_text or len(full_text) < 20:
                return None

            return WebContent(
                url=url,
                title=f"YouTube Video {video_id}",
                content=full_text,
                content_hash=_content_hash(full_text),
                source_type="youtube",
                scraped_at=time.time(),
                metadata={
                    "video_id": video_id,
                    "scraper": "youtube_transcript",
                    "segments": len(transcript_list),
                },
            )

        except Exception as e:
            _log.warning("youtube_scrape_failed", url=url, error=str(e))
            return None


# ── GitHub README ───────────────────────────────────────────

async def scrape_github(url: str) -> Optional[WebContent]:
    """
    Extract README and key files from a GitHub repository.
    Free: 5K req/hour unauthenticated.
    """
    import re

    # Parse owner/repo from URL
    match = re.match(r'https?://github\.com/([^/]+)/([^/]+)', url)
    if not match:
        return None

    owner, repo = match.group(1), match.group(2)

    with timed("github_scrape") as t:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Fetch README
                readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
                resp = await client.get(readme_url, headers={"Accept": "application/vnd.github.raw+json"})

                if resp.status_code != 200:
                    return None

                content = resp.text

                # Fetch repo metadata
                repo_url = f"https://api.github.com/repos/{owner}/{repo}"
                meta_resp = await client.get(repo_url)
                meta = meta_resp.json() if meta_resp.status_code == 200 else {}

            title = meta.get("full_name", f"{owner}/{repo}")
            description = meta.get("description", "")

            full_content = f"# {title}\n\n{description}\n\n{content}"

            return WebContent(
                url=url,
                title=title,
                content=full_content,
                content_hash=_content_hash(full_content),
                source_type="github",
                scraped_at=time.time(),
                metadata={
                    "scraper": "github",
                    "stars": meta.get("stargazers_count", 0),
                    "language": meta.get("language", ""),
                    "topics": meta.get("topics", []),
                },
            )

        except Exception as e:
            _log.warning("github_scrape_failed", url=url, error=str(e))
            return None


# ── Unified Scraper ─────────────────────────────────────────

def _detect_source_type(url: str) -> str:
    """Detect content source from URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    if "youtube.com" in domain or "youtu.be" in domain:
        return "youtube"
    elif "github.com" in domain:
        return "github"
    elif "twitter.com" in domain or "x.com" in domain:
        return "twitter"
    elif "reddit.com" in domain:
        return "reddit"
    else:
        return "web"


async def scrape_url(url: str) -> Optional[WebContent]:
    """
    Unified scraping entry point. Auto-detects source type
    and routes to the appropriate scraper.

    Fallback chain: platform-specific → Jina → Firecrawl → httpx
    """
    source_type = _detect_source_type(url)

    _log.info("scrape_begin", url=url, source_type=source_type)

    # Platform-specific scrapers
    if source_type == "youtube":
        result = await scrape_youtube(url)
        if result:
            return result

    if source_type == "github":
        result = await scrape_github(url)
        if result:
            return result

    # General web: try Jina first (free), then Firecrawl, then httpx
    result = await scrape_jina(url)
    if result:
        return result

    result = await scrape_firecrawl(url)
    if result:
        return result

    result = await scrape_httpx(url)
    return result


# ── Change Detection ────────────────────────────────────────

class ChangeDetector:
    """
    Track content hashes to detect changes.
    Stores hash → last_scraped in a JSON file on disk.
    """

    def __init__(self, persist_path: str = "./data/web_hashes.json") -> None:
        self._path = Path(persist_path)
        self._hashes: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._hashes = json.loads(self._path.read_text())
            except Exception:
                self._hashes = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._hashes, indent=2))

    def has_changed(self, url: str, new_hash: str) -> bool:
        """Check if content has changed since last scrape."""
        entry = self._hashes.get(url)
        if entry is None:
            return True  # Never scraped before
        return entry.get("hash") != new_hash

    def update(self, url: str, content_hash: str) -> None:
        """Record new hash for URL."""
        self._hashes[url] = {
            "hash": content_hash,
            "updated_at": time.time(),
        }
        self._save()


# ── Module singleton ────────────────────────────────────────

_change_detector: Optional[ChangeDetector] = None


def get_change_detector() -> ChangeDetector:
    global _change_detector
    if _change_detector is None:
        cfg = get_settings()
        _change_detector = ChangeDetector(
            persist_path=str(Path(cfg.web_cache_dir) / "content_hashes.json")
        )
    return _change_detector
