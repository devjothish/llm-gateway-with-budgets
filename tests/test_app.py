"""Does the service actually assemble?

This file exists because of a specific miss. The limiter and ledger suites both
passed while `app.main` could not be imported at all: a missing Logfire extra
made `configure_observability()` raise at import time, and nothing in the suite
imported the app, so CI would have been green on a service that cannot boot.

Any test that touches `app.main` closes that hole. These are the cheapest ones
that do.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import load_config
from app.main import app
from app.providers import CompletionRequest, Message, build_providers


def test_app_imports_and_registers_its_routes() -> None:
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert {"/v1/chat/completions", "/v1/stats", "/health"} <= paths


def test_no_credentials_registers_no_providers() -> None:
    """`lifespan` turns this into a startup failure. A gateway that boots with
    no provider authenticates, rate limits, bills nothing and answers nothing."""
    assert build_providers(None, None) == {}
    assert set(build_providers("k", None)) == {"anthropic"}
    assert set(build_providers(None, "k")) == {"openai"}
    # Ollama is credential-free, so a base URL alone is enough to enable it.
    assert set(build_providers(None, None, "http://127.0.0.1:11434/v1")) == {"ollama"}


def test_auth_runs_before_any_service_is_touched() -> None:
    """Only `config` is populated here: no Redis, no Postgres, no provider.

    If the request reaches any of them it raises `AttributeError` on the missing
    state and this returns 500, so a 401 is positive evidence that auth is the
    first thing on the path. That ordering is what keeps an unauthenticated
    caller from being able to cost anything.
    """
    app.state.config = load_config(Path("gateway.yaml"))
    client = TestClient(app)  # not a context manager: lifespan does not run
    body = {"model": "gpt", "messages": [{"role": "user", "content": "hi"}]}

    assert client.post("/v1/chat/completions", json=body).status_code == 401

    bad = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    assert bad.json()["detail"] == "invalid_api_key"


def test_request_schema_rejects_an_empty_message_list() -> None:
    """Typed at the boundary: malformed input dies in validation, not in a
    provider adapter halfway through the request path."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CompletionRequest(model="gpt", messages=[])

    ok = CompletionRequest(model="gpt", messages=[Message(role="user", content="hi")])
    assert ok.max_tokens == 1024 and ok.stream is False


def test_shipped_config_is_loadable() -> None:
    """gateway.yaml is copied into the image, so a broken one is a crash loop."""
    cfg = load_config(Path("gateway.yaml"))
    assert cfg.models and cfg.teams
    for team in cfg.teams.values():
        assert team.monthly_budget_usd > Decimal(0)
        assert set(team.allowed_models) <= cfg.models.keys()
