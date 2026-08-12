"""The gateway surface.

Request path: authenticate, rate limit, reserve budget, dispatch, settle.

Every stage emits a span, including the stages that reject. A gateway that
returns 429 without recording which limit fired and for whom is a black box,
and the first question anyone asks about a rejected request is "whose limit was
that?"

Observability is configured before the app is built so provider clients created
at import are already instrumented.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Annotated

import logfire
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response

from app.observability import configure_observability, instrument_app

configure_observability()

from app.config import GatewayConfig, ModelSpec, TeamSpec, load_config  # noqa: E402
from app.ledger import Ledger  # noqa: E402
from app.limits import RateLimiter, make_redis  # noqa: E402
from app.providers import (  # noqa: E402
    Completion,
    CompletionRequest,
    Provider,
    build_providers,
)
from app.settings import settings  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.config = load_config(settings.config_path)
    app.state.redis = make_redis(settings.redis_url)
    app.state.limiter = RateLimiter(app.state.redis)
    app.state.ledger = await Ledger.connect(settings.database_url)
    app.state.providers = build_providers(
        settings.anthropic_api_key, settings.openai_api_key, settings.ollama_base_url
    )
    if not app.state.providers:
        # Booting with no provider configured yields a service that authenticates,
        # rate limits, bills nothing and answers nothing. Fail at startup instead.
        raise RuntimeError(
            "no provider configured; set ANTHROPIC_API_KEY, OPENAI_API_KEY or OLLAMA_BASE_URL"
        )
    try:
        yield
    finally:
        await app.state.ledger.close()
        await app.state.redis.aclose()


app = FastAPI(title="llm-gateway", lifespan=lifespan)
instrument_app(app)


async def authenticate(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> TeamSpec:
    """Bearer key to team. The 401 is deliberately identical for a missing key
    and an unknown one; distinguishing them tells a prober which half to work on."""
    config: GatewayConfig = request.app.state.config
    key = authorization.removeprefix("Bearer ").strip() if authorization else ""
    team = config.team_by_key(key) if key else None
    if team is None:
        logfire.info("rejected", stage="auth", reason="unknown_key")
        raise HTTPException(status_code=401, detail="invalid_api_key")
    return team


def _estimate_input_tokens(req: CompletionRequest) -> int:
    """Characters over four. Crude, and it does not need to be better: it is a
    reservation that `settle` corrects against the provider's real count.
    ponytail: swap for the provider tokenizer if reservations drift enough to
    matter, which the reconcile spans will show before anyone guesses."""
    return sum(len(m.content) for m in req.messages) // 4


@app.post("/v1/chat/completions")
async def chat_completions(
    body: CompletionRequest,
    response: Response,
    request: Request,
    team: Annotated[TeamSpec, Depends(authenticate)],
) -> dict[str, object]:
    config: GatewayConfig = request.app.state.config
    limiter: RateLimiter = request.app.state.limiter
    ledger: Ledger = request.app.state.ledger
    providers: dict[str, Provider] = request.app.state.providers

    with logfire.span("gateway.request", team_id=team.id, model_requested=body.model) as span:
        if body.stream:
            # Honest 501 over a silently non-streamed response: a client that
            # asked for a stream and got one JSON blob will break in a way that
            # looks like a client bug.
            raise HTTPException(status_code=501, detail="streaming_not_supported")

        spec = _resolve_model(config, team, body.model)
        provider = providers.get(spec.provider)
        if provider is None:
            logfire.info("rejected", stage="route", reason="provider_unconfigured")
            raise HTTPException(status_code=503, detail=f"provider_unavailable:{spec.provider}")

        estimated_tokens = _estimate_input_tokens(body) + settings.estimated_output_tokens
        estimated_usd = spec.cost(_estimate_input_tokens(body), settings.estimated_output_tokens)

        await _enforce_rate_limit(limiter, team, estimated_tokens, response)
        await _enforce_budget(ledger, team, estimated_usd, response)

        try:
            with logfire.span("gateway.provider", provider=spec.provider, model=spec.id):
                completion = await provider.complete(body, spec)
        except Exception as exc:
            # The reservation must come back whatever went wrong, or a flapping
            # provider quietly consumes a team's budget for zero responses.
            await ledger.refund(team.id, estimated_usd=estimated_usd)
            await limiter.reconcile(
                team.id,
                now_ms=int(time.time() * 1000),
                tokens_per_minute=team.tokens_per_minute,
                delta=-estimated_tokens,
            )
            logfire.error("provider_failed", provider=spec.provider, error=str(exc))
            raise HTTPException(status_code=502, detail="provider_error") from exc

        actual_usd = spec.cost(completion.tokens_in, completion.tokens_out)
        await ledger.settle(
            team.id,
            model=spec.id,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            actual_usd=actual_usd,
            estimated_usd=estimated_usd,
        )
        await limiter.reconcile(
            team.id,
            now_ms=int(time.time() * 1000),
            tokens_per_minute=team.tokens_per_minute,
            delta=(completion.tokens_in + completion.tokens_out) - estimated_tokens,
        )

        span.set_attributes(
            {
                "tokens_in": completion.tokens_in,
                "tokens_out": completion.tokens_out,
                "cost_usd": float(actual_usd),
                "latency_ms": completion.latency_ms,
                "model_served": completion.model_id,
            }
        )
        response.headers["X-Gateway-Model"] = completion.model_id
        response.headers["X-Gateway-Cost-USD"] = f"{actual_usd:.6f}"
        return _openai_shape(completion)


def _resolve_model(config: GatewayConfig, team: TeamSpec, requested: str) -> ModelSpec:
    spec = config.models.get(requested)
    if spec is None:
        logfire.info("rejected", stage="route", reason="unknown_model", model=requested)
        raise HTTPException(status_code=404, detail=f"unknown_model:{requested}")
    if requested not in team.allowed_models:
        logfire.info("rejected", stage="route", reason="model_not_allowed", team_id=team.id)
        raise HTTPException(status_code=403, detail=f"model_not_allowed:{requested}")
    return spec


async def _enforce_rate_limit(
    limiter: RateLimiter, team: TeamSpec, estimated_tokens: int, response: Response
) -> None:
    decision = await limiter.acquire(
        team.id,
        now_ms=int(time.time() * 1000),
        requests_per_minute=team.requests_per_minute,
        tokens_per_minute=team.tokens_per_minute,
        estimated_tokens=estimated_tokens,
    )
    if decision.unsatisfiable:
        logfire.info("rejected", stage="rate_limit", reason="request_exceeds_bucket")
        raise HTTPException(status_code=413, detail="request_exceeds_token_budget")
    if not decision.allowed:
        retry_s = max(1, decision.retry_after_ms // 1000)
        logfire.info("rejected", stage="rate_limit", reason="rate_limited", team_id=team.id)
        raise HTTPException(
            status_code=429, detail="rate_limited", headers={"Retry-After": str(retry_s)}
        )
    response.headers["X-RateLimit-Remaining-Requests"] = str(decision.requests_remaining)
    response.headers["X-RateLimit-Remaining-Tokens"] = str(decision.tokens_remaining)


async def _enforce_budget(
    ledger: Ledger, team: TeamSpec, estimated_usd: Decimal, response: Response
) -> None:
    budget = await ledger.reserve(
        team.id, estimated_usd=estimated_usd, budget_usd=team.monthly_budget_usd
    )
    if not budget.allowed:
        logfire.warn("rejected", stage="budget", team_id=team.id, spent=float(budget.spent_usd))
        raise HTTPException(status_code=402, detail="monthly_budget_exhausted")
    if budget.warn:
        logfire.warn("budget_warning", team_id=team.id, remaining=float(budget.remaining_usd))
        response.headers["X-Gateway-Budget-Warning"] = "80pct"


def _openai_shape(c: Completion) -> dict[str, object]:
    """Response in the OpenAI chat-completions shape, so switching to this
    gateway is a base-URL change and nothing else."""
    return {
        "object": "chat.completion",
        "model": c.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": c.text},
                "finish_reason": c.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": c.tokens_in,
            "completion_tokens": c.tokens_out,
            "total_tokens": c.tokens_in + c.tokens_out,
        },
    }


@app.get("/v1/stats")
async def stats(
    request: Request, team: Annotated[TeamSpec, Depends(authenticate)]
) -> dict[str, object]:
    ledger: Ledger = request.app.state.ledger
    spent = await ledger.spend_summary(team.id)
    return {
        "team": team.id,
        "spent_usd": str(spent),
        "budget_usd": str(team.monthly_budget_usd),
        "remaining_usd": str(max(Decimal(0), team.monthly_budget_usd - spent)),
    }


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    return {
        "status": "ok",
        "providers": sorted(request.app.state.providers),
        "models": sorted(request.app.state.config.models),
    }
