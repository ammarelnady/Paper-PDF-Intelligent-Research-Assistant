"""Small deployment safeguards for the single-process demo server."""

from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings


class ApiProtectionMiddleware(BaseHTTPMiddleware):
    """Optional bearer-token protection and a basic per-client request limit."""

    def __init__(self, app):  # type: ignore[no-untyped-def]
        super().__init__(app)
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if not request.url.path.startswith("/api"):
            return await call_next(request)

        configured_token = settings.API_AUTH_TOKEN
        if configured_token:
            authorization = request.headers.get("authorization", "")
            if authorization != f"Bearer {configured_token}":
                return JSONResponse(
                    {"detail": "Authentication required."},
                    status_code=status.HTTP_401_UNAUTHORIZED,
                )

        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            client = request.client.host if request.client else "unknown"
            now = monotonic()
            with self._lock:
                bucket = self._requests[client]
                while bucket and now - bucket[0] >= 60:
                    bucket.popleft()
                if len(bucket) >= settings.RATE_LIMIT_PER_MINUTE:
                    return JSONResponse(
                        {"detail": "Rate limit exceeded. Try again later."},
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    )
                bucket.append(now)

        return await call_next(request)
