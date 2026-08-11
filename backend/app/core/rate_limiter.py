"""Rate Limiter for API Security."""
from __future__ import annotations

import time
from collections import defaultdict
from fastapi import HTTPException, Request, status

class SlidingWindowRateLimiter:
    """Rate limiter enforcing a maximum request count per time window per client IP."""

    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.window = 60.0
        self.client_requests: dict[str, list[float]] = defaultdict(list)

    def check_rate_limit(self, client_ip: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window

        # Clean old timestamps
        self.client_requests[client_ip] = [t for t in self.client_requests[client_ip] if t > cutoff]

        if len(self.client_requests[client_ip]) >= self.requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
            )

        self.client_requests[client_ip].append(now)

rate_limiter = SlidingWindowRateLimiter(requests_per_minute=120)
