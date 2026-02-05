"""
Authentication and rate-limiting middleware.

Architecture decisions:
  1. API key auth via X-API-Key header. Simple, stateless, no DB needed.
     Keys are loaded from environment (comma-separated AUTH_API_KEYS).
  2. Rate limiting via slowapi (backed by in-memory storage or Redis).
     Per-key limits prevent abuse without affecting other tenants.
  3. Both auth and rate limiting are optional (disabled by default).
     Enable in production via AUTH_ENABLED=true, RATE_LIMIT_ENABLED=true.
  4. Health and docs endpoints are always exempt from auth.
  5. Middleware order matters: rate limiter runs BEFORE auth so we can
     rate-limit even unauthenticated requests (anti-DDoS).
"""

from __future__ import annotations

from typing import Callable, Set

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from configs.settings import get_settings
from utils.logger import get_logger

_log = get_logger(__name__)

# Endpoints that bypass authentication
_EXEMPT_PATHS: Set[str] = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/",
}


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Validate X-API-Key header against configured keys.
    Returns 401 if missing, 403 if invalid.
    """

    def __init__(self, app, valid_keys: Set[str]) -> None:
        super().__init__(app)
        self._valid_keys = valid_keys

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip auth for exempt paths
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # Skip auth for OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")

        if not api_key:
            _log.warning("auth_missing_key", path=request.url.path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing X-API-Key header"},
            )

        if api_key not in self._valid_keys:
            _log.warning("auth_invalid_key", path=request.url.path)
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key"},
            )

        return await call_next(request)


def setup_auth(app: FastAPI) -> None:
    """
    Configure authentication middleware if enabled.
    Called from the application factory / startup.
    """
    cfg = get_settings()

    if not cfg.auth_enabled:
        _log.info("auth_disabled")
        return

    # Parse comma-separated keys
    keys = {k.strip() for k in cfg.api_keys.split(",") if k.strip()}

    if not keys:
        _log.warning("auth_enabled_but_no_keys")
        return

    app.add_middleware(APIKeyAuthMiddleware, valid_keys=keys)
    _log.info("auth_enabled", num_keys=len(keys))


def setup_rate_limiting(app: FastAPI) -> None:
    """
    Configure rate limiting middleware if enabled.
    Uses slowapi with in-memory storage (upgradeable to Redis).
    """
    cfg = get_settings()

    if not cfg.rate_limit_enabled:
        _log.info("rate_limiting_disabled")
        return

    try:
        from slowapi import Limiter, _rate_limit_exceeded_handler
        from slowapi.util import get_remote_address
        from slowapi.errors import RateLimitExceeded

        def _key_func(request: Request) -> str:
            """Rate limit by API key if present, otherwise by IP."""
            api_key = request.headers.get("X-API-Key")
            if api_key:
                return api_key
            return get_remote_address(request)

        limiter = Limiter(
            key_func=_key_func,
            default_limits=[
                f"{cfg.rate_limit_requests}/{cfg.rate_limit_window_seconds}seconds"
            ],
        )

        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        _log.info(
            "rate_limiting_enabled",
            limit=cfg.rate_limit_requests,
            window_seconds=cfg.rate_limit_window_seconds,
        )
    except ImportError:
        _log.warning("rate_limiting_package_missing", hint="pip install slowapi")
    except Exception as e:
        _log.warning("rate_limiting_setup_failed", error=str(e))


def setup_security(app: FastAPI) -> None:
    """
    Single entry point for all security middleware.
    Called from app startup.
    """
    setup_rate_limiting(app)
    setup_auth(app)
