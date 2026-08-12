"""Per-team rate limiting: two token buckets, one atomic decision.

A team has both a request/minute and a token/minute limit, and they have to be
checked together. Checking them as two round trips lets a request that is
refused for exceeding its token budget still burn a slot from the request
budget, so a team hammering large prompts is silently charged twice for one
rejection. The Lua script below computes both, then commits only if both pass.

Atomicity is the whole point. A read-modify-write in Python is a race: under
concurrency a burst slips through and the limit becomes a suggestion.

`now_ms` is supplied by the caller rather than read from Redis `TIME`. That
makes refill behaviour testable without sleeping, which is what lets the test
below assert on refill at all.
ponytail: the ceiling is clock skew between gateway instances. Sub-second skew
costs a fractional token of accuracy. If instances ever drift by seconds, move
to `redis.call('TIME')` and lose the deterministic test.
"""

from __future__ import annotations

from dataclasses import dataclass

from redis.asyncio import BlockingConnectionPool, Redis
from redis.commands.core import AsyncScript


def make_redis(url: str, *, max_connections: int = 64, timeout: float = 5.0) -> Redis:
    """A *blocking* pool, deliberately.

    redis-py's default pool raises `MaxConnectionsError` once every connection
    is checked out, which turns a traffic burst into a wave of 500s from the
    component whose entire job is to handle traffic bursts gracefully. A
    blocking pool makes exhaustion a short wait instead, and `timeout` bounds
    that wait so a genuinely stuck Redis still fails rather than hanging.

    Found by the concurrency test below, which exhausted the default pool long
    before it got near the rate limit it was meant to be testing.
    """
    pool = BlockingConnectionPool.from_url(url, max_connections=max_connections, timeout=timeout)
    return Redis(connection_pool=pool)


# Both buckets are refilled and evaluated before either is written.
# KEYS:  1=request bucket  2=token bucket
# ARGV:  1=now_ms  2=req_cap  3=req_per_ms  4=tok_cap  5=tok_per_ms  6=tok_cost  7=ttl_s
# Returns {allowed, retry_after_ms, req_remaining, tok_remaining}
_ACQUIRE_LUA = """
local function refill(key, cap, rate, now)
  local h = redis.call('HMGET', key, 't', 'ts')
  local tokens = tonumber(h[1])
  local ts = tonumber(h[2])
  if tokens == nil then return cap, now end
  return math.min(cap, tokens + (now - ts) * rate), now
end

local now      = tonumber(ARGV[1])
local req_cap  = tonumber(ARGV[2])
local req_rate = tonumber(ARGV[3])
local tok_cap  = tonumber(ARGV[4])
local tok_rate = tonumber(ARGV[5])
local tok_cost = tonumber(ARGV[6])
local ttl      = tonumber(ARGV[7])

local req, _ = refill(KEYS[1], req_cap, req_rate, now)
local tok, _ = refill(KEYS[2], tok_cap, tok_rate, now)

-- A single request costing more than the entire bucket can never succeed.
-- Refusing it as unsatisfiable beats reporting a retry_after that will not help.
if tok_cost > tok_cap then
  return {0, -1, math.floor(req), math.floor(tok)}
end

if req < 1 or tok < tok_cost then
  local wait_req = 0
  local wait_tok = 0
  if req < 1 then wait_req = (1 - req) / req_rate end
  if tok < tok_cost then wait_tok = (tok_cost - tok) / tok_rate end
  local wait = math.ceil(math.max(wait_req, wait_tok))
  return {0, wait, math.floor(req), math.floor(tok)}
end

redis.call('HSET', KEYS[1], 't', req - 1, 'ts', now)
redis.call('HSET', KEYS[2], 't', tok - tok_cost, 'ts', now)
redis.call('EXPIRE', KEYS[1], ttl)
redis.call('EXPIRE', KEYS[2], ttl)
return {1, 0, math.floor(req - 1), math.floor(tok - tok_cost)}
"""

# Settles the estimate made at admission against what the response actually
# cost. A positive delta overdraws the bucket, which is deliberate: the next
# request from a team that badly under-estimated should wait for it.
# KEYS: 1=token bucket   ARGV: 1=now_ms 2=cap 3=per_ms 4=delta 5=ttl_s
_RECONCILE_LUA = """
local now   = tonumber(ARGV[1])
local cap   = tonumber(ARGV[2])
local rate  = tonumber(ARGV[3])
local delta = tonumber(ARGV[4])
local ttl   = tonumber(ARGV[5])

local h = redis.call('HMGET', KEYS[1], 't', 'ts')
local tokens = tonumber(h[1])
local ts = tonumber(h[2])
if tokens == nil then tokens = cap; ts = now end

tokens = math.min(cap, tokens + (now - ts) * rate) - delta
-- Floor the debt at one full bucket so a single pathological response cannot
-- lock a team out for hours.
if tokens < -cap then tokens = -cap end

redis.call('HSET', KEYS[1], 't', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], ttl)
return math.floor(tokens)
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after_ms: int
    requests_remaining: int
    tokens_remaining: int

    @property
    def unsatisfiable(self) -> bool:
        """The request can never succeed: it costs more than the bucket holds.
        Callers should return 413, not 429 with a Retry-After that lies."""
        return not self.allowed and self.retry_after_ms < 0


class RateLimiter:
    def __init__(self, redis: Redis) -> None:
        self._acquire: AsyncScript = redis.register_script(_ACQUIRE_LUA)
        self._reconcile: AsyncScript = redis.register_script(_RECONCILE_LUA)

    @staticmethod
    def _ttl_seconds(capacity: float, per_ms: float) -> int:
        """Long enough for a drained bucket to refill completely, so an idle
        team's key never expires mid-refill and hands back a free full bucket."""
        return max(60, int((capacity / per_ms) / 1000) * 2)

    async def acquire(
        self,
        team_id: str,
        *,
        now_ms: int,
        requests_per_minute: int,
        tokens_per_minute: int,
        estimated_tokens: int,
    ) -> Decision:
        req_rate = requests_per_minute / 60_000
        tok_rate = tokens_per_minute / 60_000
        result = await self._acquire(
            keys=[f"rl:{team_id}:req", f"rl:{team_id}:tok"],
            args=[
                now_ms,
                requests_per_minute,
                req_rate,
                tokens_per_minute,
                tok_rate,
                estimated_tokens,
                self._ttl_seconds(tokens_per_minute, tok_rate),
            ],
        )
        allowed, retry_ms, req_left, tok_left = (int(x) for x in result)
        return Decision(bool(allowed), retry_ms, req_left, tok_left)

    async def reconcile(
        self, team_id: str, *, now_ms: int, tokens_per_minute: int, delta: int
    ) -> None:
        """`delta` = actual tokens - estimate. Negative refunds an over-estimate."""
        if delta == 0:
            return
        tok_rate = tokens_per_minute / 60_000
        await self._reconcile(
            keys=[f"rl:{team_id}:tok"],
            args=[
                now_ms,
                tokens_per_minute,
                tok_rate,
                delta,
                self._ttl_seconds(tokens_per_minute, tok_rate),
            ],
        )
