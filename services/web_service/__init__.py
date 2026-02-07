"""
Web Service Package — web scraping, crawling, and content ingestion.
"""

from services.web_service.web_scraper import WebContent, ChangeDetector, get_change_detector

__all__ = [
    "WebContent",
    "ChangeDetector",
    "get_change_detector",
]
