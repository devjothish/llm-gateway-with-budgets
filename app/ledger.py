"""Spend tracking and budget enforcement, in Postgres.

Two tables, because they answer different questions. `spend_ledger` is
append-only and answers "what did this team actually spend it on" — the audit
trail. `team_spend` is a per-period counter and answers "can this request run"
in one indexed read, so enforcement does not scan the ledger on every call.

Postgres rather than a Redis counter: this is money, and a counter that a
restart forgets converts a budget cap into a suggestion.

Enforcement uses the same reserve-then-settle shape as the rate limiter. Cost
is not knowable until the response exists, so admission charges an estimate
atomically, and `settle` applies the difference. Checking first and charging
afterwards would let concurrent requests each read 99% of budget and all pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import asyncpg

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend_ledger (
    id          BIGSERIAL PRIMARY KEY,
    team_id     TEXT        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    model       TEXT        NOT NULL,
    tokens_in   INTEGER     NOT NULL,
    tokens_out  INTEGER     NOT NULL,
    cost_usd    NUMERIC(12, 6) NOT NULL
);
CREATE INDEX IF NOT EXISTS spend_ledger_team_ts ON spend_ledger (team_id, ts DESC);

CREATE TABLE IF NOT EXISTS team_spend (
    team_id    TEXT           NOT NULL,
    period     TEXT           NOT NULL,
    spent_usd  NUMERIC(12, 6) NOT NULL DEFAULT 0,
    PRIMARY KEY (team_id, period)
);
"""

# Atomic increment returning the post-increment total. The read and the write
# are one statement, so two concurrent requests cannot both see the same
# pre-spend total and both be admitted.
_CHARGE = """
INSERT INTO team_spend (team_id, period, spent_usd)
VALUES ($1, $2, $3)
ON CONFLICT (team_id, period)
DO UPDATE SET spent_usd = team_spend.spent_usd + EXCLUDED.spent_usd
RETURNING spent_usd
"""


def current_period(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m")


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    spent_usd: Decimal
    budget_usd: Decimal

    @property
    def remaining_usd(self) -> Decimal:
        return max(Decimal(0), self.budget_usd - self.spent_usd)

    @property
    def warn(self) -> bool:
        """80% is where a team should hear about it, not 100%."""
        return self.budget_usd > 0 and self.spent_usd / self.budget_usd >= Decimal("0.8")


class Ledger:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> Ledger:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
        if pool is None:  # pragma: no cover - asyncpg types this as Optional
            raise RuntimeError(f"could not create a connection pool for {dsn!r}")
        async with pool.acquire() as conn:
            # ponytail: CREATE TABLE IF NOT EXISTS at boot. Fine for one service
            # owning one schema; swap for Alembic the first time a column changes
            # shape rather than gets added.
            await conn.execute(_SCHEMA)
        return cls(pool)

    async def close(self) -> None:
        await self._pool.close()

    async def reserve(
        self, team_id: str, *, estimated_usd: Decimal, budget_usd: Decimal
    ) -> BudgetDecision:
        """Charge an estimate up front. On refusal the estimate is refunded, so a
        rejected request leaves the counter exactly where it started."""
        period = current_period()
        async with self._pool.acquire() as conn, conn.transaction():
            spent: Decimal = await conn.fetchval(_CHARGE, team_id, period, estimated_usd)
            if spent > budget_usd:
                await conn.fetchval(_CHARGE, team_id, period, -estimated_usd)
                return BudgetDecision(False, spent - estimated_usd, budget_usd)
            return BudgetDecision(True, spent, budget_usd)

    async def settle(
        self,
        team_id: str,
        *,
        model: str,
        tokens_in: int,
        tokens_out: int,
        actual_usd: Decimal,
        estimated_usd: Decimal,
    ) -> None:
        """Write the audit row and correct the counter by the estimate's error."""
        period = current_period()
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "INSERT INTO spend_ledger (team_id, model, tokens_in, tokens_out, cost_usd)"
                " VALUES ($1, $2, $3, $4, $5)",
                team_id,
                model,
                tokens_in,
                tokens_out,
                actual_usd,
            )
            delta = actual_usd - estimated_usd
            if delta != 0:
                await conn.fetchval(_CHARGE, team_id, period, delta)

    async def refund(self, team_id: str, *, estimated_usd: Decimal) -> None:
        """Called when the provider call fails. A request that produced no
        response bills nothing, or a flapping provider silently eats a budget."""
        await self._pool.fetchval(_CHARGE, team_id, current_period(), -estimated_usd)

    async def spend_summary(self, team_id: str) -> Decimal:
        period = current_period()
        value: Decimal | None = await self._pool.fetchval(
            "SELECT spent_usd FROM team_spend WHERE team_id = $1 AND period = $2",
            team_id,
            period,
        )
        return value or Decimal(0)
