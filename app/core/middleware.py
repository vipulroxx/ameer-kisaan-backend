"""Security headers, request IDs, body-size limits, rate limiting and JSON logs."""
import json
import logging
import sys
import threading
import time
import uuid
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import get_settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name,
                "msg": record.getMessage()}
        for key in ("request_id", "path", "method", "status", "duration_ms", "user_id", "action"):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def client_ip(request: Request) -> str:
    """The address of whoever connected to our edge proxy.

    Clients can send their own X-Forwarded-For header, so the leftmost entries
    are attacker-controlled. Our single trusted proxy (the AWS ALB, or nginx in
    docker-compose) appends the real peer address last, so we take the
    rightmost entry (TRUSTED_PROXY_HOPS=1). Behind more proxies, raise
    TRUSTED_PROXY_HOPS or name a header the edge sets (CLIENT_IP_HEADER).
    Never trust this header if the API is exposed directly.
    """
    s = get_settings()
    if s.client_ip_header:
        # A header our edge sets itself and overwrites on every request (e.g. Cloudflare's
        # CF-Connecting-IP / True-Client-IP in front of Render), so clients cannot spoof it.
        ip = request.headers.get(s.client_ip_header)
        if ip:
            return ip.strip()
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        hops = [h.strip() for h in fwd.split(",") if h.strip()]
        if hops:
            # TRUSTED_PROXY_HOPS proxies each append one entry; the client is that many from the right.
            return hops[-min(max(s.trusted_proxy_hops, 1), len(hops))]
    return request.client.host if request.client else "unknown"


# Content-Security-Policy for the React app when the API serves it (STATIC_DIR).
# Photos are same-origin (PHOTO_URL_MODE=proxy); https: covers weather icons and map tiles.
WEB_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://unpkg.com; "
           "img-src 'self' data: blob: https:; font-src 'self' data: https://fonts.gstatic.com; "
           "connect-src 'self' https://api.open-meteo.com https://tfhub.dev https://www.kaggle.com "
           "https://*.kaggle.com https://storage.googleapis.com; frame-src https://www.youtube.com; "
           "media-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; form-action 'self'; "
           "frame-ancestors 'none'")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()

        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > get_settings().max_request_mb * 1024 * 1024:
            return JSONResponse({"detail": "Request body too large."}, status_code=413)

        response: Response = await call_next(request)
        s = get_settings()
        path = request.url.path
        is_api = path.startswith((s.api_prefix, "/health", "/docs", "/redoc", "/openapi.json"))
        h = response.headers
        h["X-Request-ID"] = request_id
        h["X-Content-Type-Options"] = "nosniff"
        h["X-Frame-Options"] = "DENY"
        h["Cross-Origin-Opener-Policy"] = "same-origin"
        if s.static_dir and not is_api:
            # The web app (same policy as deploy/security-headers.conf for nginx).
            h["Referrer-Policy"] = "strict-origin-when-cross-origin"
            h["Permissions-Policy"] = "geolocation=(self), microphone=(self), camera=(self)"
            h["Content-Security-Policy"] = WEB_CSP
        else:
            h["Referrer-Policy"] = "no-referrer"
            h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            if not path.startswith(("/docs", "/redoc", "/openapi.json")):
                h["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        if path.startswith(s.api_prefix):
            h["Cache-Control"] = "no-store"
        if get_settings().is_prod:
            h["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"

        logging.getLogger("access").info(
            "request", extra={"request_id": request_id, "path": request.url.path, "method": request.method,
                              "status": response.status_code,
                              "duration_ms": round((time.perf_counter() - start) * 1000, 1)})
        return response


class RateLimiter:
    """Sliding-window limiter, per key, held in memory.

    This protects each API task. In AWS an AWS WAF rate-based rule in front of
    the load balancer (see infra/terraform/waf.tf) enforces the global limit.
    """

    def __init__(self):
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_s: int) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    def reset(self):
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()
