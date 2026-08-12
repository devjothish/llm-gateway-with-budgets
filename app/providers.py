"""Provider adapters. Vendor response objects stop here.

The gateway's contract with everything downstream is `Completion`. No route,
ledger write, or span reads a vendor SDK type, so adding a provider is a new
adapter and nothing else. This is the workspace's typed-at-the-boundary rule
applied at the only boundary this service has.

Written against the vendor SDKs directly rather than a normalization library.
Three providers need roughly a hundred lines here, and the gateway's value is
the control plane around them, not the adapters themselves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Protocol

from anthropic import AsyncAnthropic
from anthropic import omit as anthropic_omit
from anthropic.types import MessageParam
from openai import AsyncOpenAI
from openai import omit as openai_omit
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from app.config import ModelSpec

Role = Literal["system", "user", "assistant"]


class Message(BaseModel, frozen=True):
    role: Role
    content: str = Field(min_length=1)


class CompletionRequest(BaseModel):
    """The OpenAI chat-completions subset the gateway accepts.

    Deliberately a subset. Mirroring the full schema would imply support for
    parameters that are silently dropped on the way to Anthropic, and a
    parameter that is accepted and ignored is worse than one that is rejected.
    """

    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int = Field(default=1024, gt=0, le=32_000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = False


@dataclass(frozen=True)
class Completion:
    """Normalized response. Every field here is one a caller or the ledger needs."""

    text: str
    model_id: str
    tokens_in: int
    tokens_out: int
    finish_reason: str
    latency_ms: float


class Provider(Protocol):
    async def complete(self, req: CompletionRequest, spec: ModelSpec) -> Completion: ...


def _split_system(messages: list[Message]) -> tuple[str | None, list[MessageParam]]:
    """Anthropic takes the system prompt as a top-level argument, OpenAI takes it
    as the first message. Normalizing here keeps that difference out of routing."""
    system = "\n\n".join(m.content for m in messages if m.role == "system") or None
    turns: list[MessageParam] = [
        {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
        for m in messages
        if m.role != "system"
    ]
    return system, turns


class AnthropicProvider:
    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(self, req: CompletionRequest, spec: ModelSpec) -> Completion:
        system, turns = _split_system(req.messages)
        started = time.perf_counter()
        resp = await self._client.messages.create(
            model=spec.id,
            max_tokens=req.max_tokens,
            messages=turns,
            system=system if system is not None else anthropic_omit,
            temperature=req.temperature if req.temperature is not None else anthropic_omit,
        )
        elapsed = (time.perf_counter() - started) * 1000
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(
            text=text,
            model_id=resp.model,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            finish_reason=resp.stop_reason or "unknown",
            latency_ms=elapsed,
        )


class OpenAIProvider:
    """Also serves Ollama, which speaks the OpenAI chat-completions protocol.

    A third adapter would have been a copy of this one with a different host.
    The differences that do exist are the server's, not the protocol's: Ollama
    accepts `max_completion_tokens` and then ignores it, so the cap is advisory
    there. The gateway reconciles against reported usage either way, which is
    why that does not corrupt the accounting.
    """

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, req: CompletionRequest, spec: ModelSpec) -> Completion:
        messages: list[ChatCompletionMessageParam] = []
        for m in req.messages:
            if m.role == "system":
                messages.append({"role": "system", "content": m.content})
            elif m.role == "assistant":
                messages.append({"role": "assistant", "content": m.content})
            else:
                messages.append({"role": "user", "content": m.content})

        started = time.perf_counter()
        resp = await self._client.chat.completions.create(
            model=spec.id,
            # `max_completion_tokens`, not `max_tokens`: the latter is deprecated
            # and rejected outright by reasoning models.
            max_completion_tokens=req.max_tokens,
            messages=messages,
            temperature=req.temperature if req.temperature is not None else openai_omit,
        )
        elapsed = (time.perf_counter() - started) * 1000
        choice = resp.choices[0]
        # Usage is Optional in the schema. Defaulting to 0 would bill the team
        # nothing for a real call, so a missing usage block is a failure rather
        # than a free request.
        usage = _require_usage(resp.usage)
        return Completion(
            text=choice.message.content or "",
            model_id=resp.model,
            tokens_in=usage.prompt_tokens,
            tokens_out=usage.completion_tokens,
            finish_reason=choice.finish_reason or "unknown",
            latency_ms=elapsed,
        )


class _Usage(Protocol):
    prompt_tokens: int
    completion_tokens: int


def _require_usage(usage: _Usage | None) -> _Usage:
    if usage is None:
        raise ProviderError("provider returned no usage block; refusing to bill 0 tokens")
    return usage


class ProviderError(RuntimeError):
    """Raised for a response the adapter cannot normalize. Distinct from the
    SDKs' own errors so the route can tell 'provider misbehaved' from
    'provider unreachable'."""


def build_providers(
    anthropic_key: str | None,
    openai_key: str | None,
    ollama_base_url: str | None = None,
) -> dict[str, Provider]:
    """Only providers with credentials are registered. A team routed to an
    unconfigured provider gets a clear 503 rather than an SDK auth error.

    Ollama needs no key, so it is registered whenever a base URL is set. Its
    client still gets a placeholder because the OpenAI SDK requires one and
    refuses to construct without it; Ollama ignores the value.
    """
    providers: dict[str, Provider] = {}
    if anthropic_key:
        providers["anthropic"] = AnthropicProvider(anthropic_key)
    if openai_key:
        providers["openai"] = OpenAIProvider(openai_key)
    if ollama_base_url:
        providers["ollama"] = OpenAIProvider("unused-by-ollama", base_url=ollama_base_url)
    return providers
