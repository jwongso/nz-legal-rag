"""
Semaphore-based queue with per-IP rate limiting.
Max 1 concurrent LLM call; each IP limited to 1 in-flight request.
Requests wait up to 60s before receiving a 503.
"""

import asyncio

from fastapi import HTTPException, Request

_MAX_CONCURRENT = 1
_MAX_WAIT = 60.0
_AVG_QUERY_SECONDS = 25

_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
_active: int = 0
_waiting: int = 0
_ip_in_flight: dict[str, int] = {}


def get_client_ip(request: Request) -> str:
    cf = request.headers.get("CF-Connecting-IP")
    if cf:
        return cf
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host or "unknown"


def queue_status() -> dict:
    return {
        "active": _active,
        "waiting": _waiting,
        "estimated_wait_seconds": max(0, (_waiting * _AVG_QUERY_SECONDS) // max(1, _MAX_CONCURRENT)),
    }


def will_wait() -> bool:
    """Return True if the next acquire() call will block (all slots are active)."""
    return _active >= _MAX_CONCURRENT


def queue_wait_estimate() -> dict:
    """Return position and estimated wait for a request about to acquire."""
    position = _waiting + 1
    estimated = (position * _AVG_QUERY_SECONDS) // max(1, _MAX_CONCURRENT)
    return {"position": position, "active": _active, "estimated_wait_s": estimated}


async def acquire(request: Request) -> str:
    global _active, _waiting
    ip = get_client_ip(request)

    # Use .get() to avoid creating a dict entry for every unique IP that checks in.
    if _ip_in_flight.get(ip, 0) >= 1:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "You already have a query in progress. Please wait for it to finish.",
                "retry_after": 30,
            },
        )

    _waiting += 1
    _ip_in_flight[ip] = _ip_in_flight.get(ip, 0) + 1

    try:
        await asyncio.wait_for(_semaphore.acquire(), timeout=_MAX_WAIT)
    except asyncio.TimeoutError:
        _waiting -= 1
        count = _ip_in_flight.get(ip, 1) - 1
        if count <= 0:
            _ip_in_flight.pop(ip, None)
        else:
            _ip_in_flight[ip] = count
        raise HTTPException(
            status_code=503,
            detail={
                "error": "The server is busy right now. Please try again in a moment.",
                "retry_after": 30,
            },
        )

    _waiting -= 1
    _active += 1
    return ip


def release(ip: str) -> None:
    global _active
    _semaphore.release()
    _active -= 1
    count = _ip_in_flight.get(ip, 1) - 1
    if count <= 0:
        _ip_in_flight.pop(ip, None)
    else:
        _ip_in_flight[ip] = count
