"""Model prices and team policy, loaded from YAML.

Prices live here and never in code. Provider pricing changes, and a stale
constant compiled into a module silently corrupts every cost number the gateway
reports — including the ones the README will quote as results.

Money is `Decimal` end to end. Budget enforcement decides whether a request is
refused, so binary float error is not an acceptable rounding story.
"""

from __future__ import annotations

import hashlib
import hmac
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, Field, model_validator

Tier = Literal["local", "cheap", "mid", "frontier"]
Provider = Literal["anthropic", "openai", "ollama"]

_PER_MTOK = Decimal(1_000_000)


class ModelSpec(BaseModel, frozen=True):
    """One routable model. `id` is the provider's own identifier, sent verbatim."""

    id: str
    provider: Provider
    tier: Tier
    price_in_per_mtok: Decimal = Field(ge=0)
    price_out_per_mtok: Decimal = Field(ge=0)

    def cost(self, tokens_in: int, tokens_out: int) -> Decimal:
        return (
            self.price_in_per_mtok * tokens_in + self.price_out_per_mtok * tokens_out
        ) / _PER_MTOK


class TeamSpec(BaseModel, frozen=True):
    """One caller. Limits are per-team because a shared limit is not a limit.

    `key_sha256` rather than the key itself: this file is committed, and a
    gateway that leaks its own client credentials is a worse story than no
    gateway. Generate with `python -m app.config hash <key>`.
    """

    id: str
    key_sha256: str = Field(min_length=64, max_length=64)
    requests_per_minute: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)
    monthly_budget_usd: Decimal = Field(gt=0)
    allowed_models: list[str] = Field(min_length=1)


class GatewayConfig(BaseModel, frozen=True):
    models: dict[str, ModelSpec]
    teams: dict[str, TeamSpec]

    @model_validator(mode="after")
    def _every_allowed_model_exists(self) -> Self:
        """A typo in `allowed_models` would otherwise surface as a 403 at 3am
        rather than a startup failure."""
        for team in self.teams.values():
            unknown = set(team.allowed_models) - self.models.keys()
            if unknown:
                raise ValueError(f"team {team.id!r} allows unknown models: {sorted(unknown)}")
        return self

    def team_by_key(self, api_key: str) -> TeamSpec | None:
        digest = hashlib.sha256(api_key.encode()).hexdigest()
        # Linear scan: the team count is small and bounded by a committed file.
        # ponytail: dict keyed on digest if this ever holds thousands of teams.
        return next(
            (t for t in self.teams.values() if hmac.compare_digest(t.key_sha256, digest)), None
        )


def load_config(path: Path) -> GatewayConfig:
    raw = yaml.safe_load(path.read_text())
    # The YAML key is the default id; an explicit `id:` in the body wins, which
    # is how a short alias ("gpt") maps to a provider's real identifier.
    models = {k: ModelSpec(**{"id": k, **v}) for k, v in raw["models"].items()}
    teams = {k: TeamSpec(**{"id": k, **v}) for k, v in raw["teams"].items()}
    return GatewayConfig(models=models, teams=teams)


if __name__ == "__main__":  # key-hashing helper; see TeamSpec.key_sha256
    import sys

    print(hashlib.sha256(sys.argv[1].encode()).hexdigest())
