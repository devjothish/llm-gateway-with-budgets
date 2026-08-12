"""Budget enforcement against a real Postgres.

`docker compose up -d postgres` then `pytest tests/ -q`. Skips if absent.

The invariant worth guarding is the one that is easy to get wrong and expensive
when it is: a refused request must leave the spend counter exactly where it
started. Reserve-then-refund that does not fully refund converts every rejection
into a slow leak against the team's budget, and nothing surfaces it until the
month closes short.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from decimal import Decimal

import pytest

from app.ledger import Ledger, current_period

DSN = os.getenv("DATABASE_URL", "postgresql://gateway:gateway@localhost:5433/gateway")


@pytest.fixture
async def ledger() -> Ledger:
    try:
        return await Ledger.connect(DSN)
    except OSError:
        # Only "nothing is listening" skips. A bad role or a wrong database is a
        # misconfiguration and must fail: catching everything here once turned an
        # auth error into seven green skips, which reads exactly like a pass.
        pytest.skip(f"no Postgres reachable at {DSN}; `docker compose up -d postgres`")


def team() -> str:
    return f"t-{uuid.uuid4().hex[:12]}"


async def test_reserve_within_budget_is_allowed_and_counted(ledger: Ledger) -> None:
    t = team()
    d = await ledger.reserve(t, estimated_usd=Decimal("1.50"), budget_usd=Decimal("10.00"))
    assert d.allowed
    assert d.spent_usd == Decimal("1.50")
    assert d.remaining_usd == Decimal("8.50")


async def test_refused_reservation_leaves_the_counter_untouched(ledger: Ledger) -> None:
    """The leak this suite exists to prevent."""
    t = team()
    budget = Decimal("10.00")

    await ledger.reserve(t, estimated_usd=Decimal("9.00"), budget_usd=budget)
    before = await ledger.spend_summary(t)
    assert before == Decimal("9.00")

    refused = await ledger.reserve(t, estimated_usd=Decimal("5.00"), budget_usd=budget)
    assert not refused.allowed

    after = await ledger.spend_summary(t)
    assert after == before, f"refused reservation moved the counter {before} -> {after}"


async def test_settle_corrects_the_estimate_and_writes_the_audit_row(ledger: Ledger) -> None:
    t = team()
    estimate = Decimal("0.010000")
    actual = Decimal("0.025000")

    await ledger.reserve(t, estimated_usd=estimate, budget_usd=Decimal("10.00"))
    await ledger.settle(
        t,
        model="claude-sonnet-5",
        tokens_in=1000,
        tokens_out=500,
        actual_usd=actual,
        estimated_usd=estimate,
    )

    assert await ledger.spend_summary(t) == actual, (
        "counter should hold the actual, not the estimate"
    )

    row = await ledger._pool.fetchrow(  # noqa: SLF001 - asserting on the audit trail
        "SELECT model, tokens_in, tokens_out, cost_usd FROM spend_ledger WHERE team_id = $1", t
    )
    assert row is not None, "settle wrote no audit row"
    assert row["model"] == "claude-sonnet-5"
    assert (row["tokens_in"], row["tokens_out"]) == (1000, 500)
    assert row["cost_usd"] == actual


async def test_settle_refunds_an_overestimate(ledger: Ledger) -> None:
    """Reserving high and spending low must give the difference back, or every
    short response permanently overcharges the team."""
    t = team()
    await ledger.reserve(t, estimated_usd=Decimal("1.00"), budget_usd=Decimal("10.00"))
    await ledger.settle(
        t,
        model="m",
        tokens_in=1,
        tokens_out=1,
        actual_usd=Decimal("0.10"),
        estimated_usd=Decimal("1.00"),
    )
    assert await ledger.spend_summary(t) == Decimal("0.10")


async def test_refund_returns_a_failed_reservation(ledger: Ledger) -> None:
    t = team()
    await ledger.reserve(t, estimated_usd=Decimal("2.00"), budget_usd=Decimal("10.00"))
    await ledger.refund(t, estimated_usd=Decimal("2.00"))
    assert await ledger.spend_summary(t) == Decimal("0.00")


async def test_concurrent_reserves_cannot_both_pass_the_boundary(ledger: Ledger) -> None:
    """Two requests arriving together at 99% of budget must not both be admitted.

    A check-then-charge implementation passes every test above and fails this
    one: both reads see the same pre-spend total.
    """
    t = team()
    budget = Decimal("10.00")
    await ledger.reserve(t, estimated_usd=Decimal("9.00"), budget_usd=budget)

    results = await asyncio.gather(
        *(ledger.reserve(t, estimated_usd=Decimal("0.60"), budget_usd=budget) for _ in range(20))
    )
    admitted = sum(1 for r in results if r.allowed)

    # 1.00 of headroom, 0.60 per request: exactly one fits.
    assert admitted == 1, f"{admitted} requests admitted against 1.00 of headroom at 0.60 each"
    assert await ledger.spend_summary(t) == Decimal("9.60")


async def test_warn_fires_at_eighty_percent(ledger: Ledger) -> None:
    t = team()
    d = await ledger.reserve(t, estimated_usd=Decimal("8.00"), budget_usd=Decimal("10.00"))
    assert d.allowed and d.warn

    quiet = await ledger.reserve(team(), estimated_usd=Decimal("1.00"), budget_usd=Decimal("10.00"))
    assert quiet.allowed and not quiet.warn


async def test_period_key_is_year_month() -> None:
    from datetime import UTC, datetime

    assert current_period(datetime(2026, 8, 11, tzinfo=UTC)) == "2026-08"
