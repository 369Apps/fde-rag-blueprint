"""Request tracing middleware: every request gets an id, timing, and a
structured log line. In production, ship these to your observability stack;
the shape here matches what a client SRE team expects to consume.
"""
from __future__ import annotations

import json
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        print(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round(elapsed_ms, 2),
                }
            ),
            flush=True,
        )
        return response
