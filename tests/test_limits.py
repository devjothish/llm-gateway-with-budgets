"""Token bucket behaviour, against a real Redis because the logic is in Lua.

`docker compose up -d redis` then `pytest tests/ -q`. Skips if Redis is absent
rather than failing, so `ruff`/`mypy`/unit runs stay usable without services.

The concurrency case is the one that matters most: it is the test that goes red
if someone "simplifies" the Lua into a read-modify-write in Python.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.limits import RateLimiter, make_redis


@pytest.fixture
async def limiter() -> RateLimiter:
    # Built through `make_redis` rather than a bare client, so the tests
    # exercise the pool configuration the gateway actually runs with.
    url = os.getenv("REDIS_URL", "redis://localhost:6379/15")
    redis = make_redis(url)
    try:
        await redis.ping()
    except (OSError, RedisConnectionError):
        # Narrow on purpose: only an unreachable Redis skips. Anything else is a
        # misconfiguration, and a skip that looks like a pass is how a suite
        # stops testing without anyone noticing.
        pytest.skip(f"no Redis reachable at {url}; `docker compose up -d redis`")
    return RateLimiter(redis)


def team() -> str:
    """Fresh id per test. Shared ids would couple tests through Redis state."""
    return f"t-{uuid.uuid4().hex[:12]}"


async def test_burst_up_to_capacity_then_denied(limiter: RateLimiter) -> None:
    t = team()
    for i in range(5):
        d = await limiter.acquire(
            t, now_ms=0, requests_per_minute=5, tokens_per_minute=10_000, estimated_tokens=10
        )
        assert d.allowed, f"request {i} should fit in a capacity-5 bucket"

    d = await limiter.acquire(
        t, now_ms=0, requests_per_minute=5, tokens_per_minute=10_000, estimated_tokens=10
    )
    assert not d.allowed
    assert d.retry_after_ms > 0


async def test_token_denial_does_not_consume_a_request_slot(limiter: RateLimiter) -> None:
    """The reason both buckets live in one script.

    A request too large for the token bucket must be refused without spending
    from the request bucket, or a team sending oversized prompts is charged
    twice for one rejection.
    """
    t = team()
    args = {"requests_per_minute": 10, "tokens_per_minute": 1_000}

    denied = await limiter.acquire(t, now_ms=0, estimated_tokens=900, **args)
    assert denied.allowed
    denied = await limiter.acquire(t, now_ms=0, estimated_tokens=900, **args)
    assert not denied.allowed, "second 900-token request exceeds the 1000-token bucket"

    # One request was admitted, so exactly one request slot should be gone.
    d = await limiter.acquire(t, now_ms=0, estimated_tokens=1, **args)
    assert d.allowed
    assert d.requests_remaining == 8, (
        f"expected 8 request slots left (10 - 2 admitted), got {d.requests_remaining}; "
        "the token-bucket denial leaked a request slot"
    )


async def test_refills_over_time(limiter: RateLimiter) -> None:
    t = team()
    args = {"requests_per_minute": 60, "tokens_per_minute": 60_000}

    for _ in range(60):
        assert (await limiter.acquire(t, now_ms=0, estimated_tokens=1, **args)).allowed
    assert not (await limiter.acquire(t, now_ms=0, estimated_tokens=1, **args)).allowed

    # 60 rpm = 1 request per 1000ms. Half a second is not enough; a full one is.
    assert not (await limiter.acquire(t, now_ms=500, estimated_tokens=1, **args)).allowed
    assert (await limiter.acquire(t, now_ms=1_000, estimated_tokens=1, **args)).allowed


async def test_request_larger_than_bucket_is_unsatisfiable_not_retryable(
    limiter: RateLimiter,
) -> None:
    """Waiting cannot help, so the caller must not be handed a Retry-After."""
    d = await limiter.acquire(
        team(), now_ms=0, requests_per_minute=10, tokens_per_minute=1_000, estimated_tokens=5_000
    )
    assert not d.allowed
    assert d.unsatisfiable
    assert d.retry_after_ms == -1


async def test_reconcile_charges_the_underestimate(limiter: RateLimiter) -> None:
    t = team()
    args = {"requests_per_minute": 100, "tokens_per_minute": 1_000}

    d = await limiter.acquire(t, now_ms=0, estimated_tokens=100, **args)
    assert d.tokens_remaining == 900

    # Response actually cost 400, not the 100 reserved.
    await limiter.reconcile(t, now_ms=0, tokens_per_minute=1_000, delta=300)

    d = await limiter.acquire(t, now_ms=0, estimated_tokens=1, **args)
    assert d.tokens_remaining == 599, "the 300-token overrun was not charged"


async def test_concurrent_acquires_never_exceed_capacity(limiter: RateLimiter) -> None:
    """Atomicity. A read-modify-write in Python passes every test above and
    fails this one."""
    t = team()
    capacity = 20
    attempts = 200

    results = await asyncio.gather(
        *(
            limiter.acquire(
                t,
                now_ms=0,
                requests_per_minute=capacity,
                tokens_per_minute=10_000_000,
                estimated_tokens=1,
            )
            for _ in range(attempts)
        )
    )
    granted = sum(1 for r in results if r.allowed)
    assert granted == capacity, f"{granted} of {attempts} granted against a capacity of {capacity}"
