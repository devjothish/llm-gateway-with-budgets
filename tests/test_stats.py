"""The Wilson interval is copied from horizon-bench, so it gets checked here.

A copied function that nobody re-tests is how a transcription slip ships. These
cases pin the properties the cache sweep actually relies on.
"""

from __future__ import annotations

from bench.stats import wilson


def test_bounds_stay_inside_zero_one_at_the_extremes() -> None:
    """The reason Wilson is used instead of Wald. The cache sweep has a
    0-of-20 row and a 6-of-6 row; Wald returns bounds outside [0, 1] on both."""
    for k, n in [(0, 20), (20, 20), (0, 1), (1, 1), (6, 6)]:
        lo, hi = wilson(k, n)
        assert 0.0 <= lo <= hi <= 1.0, f"wilson({k},{n}) = ({lo},{hi}) escaped [0,1]"


def test_interval_brackets_the_point_estimate() -> None:
    for k, n in [(1, 10), (5, 10), (9, 10), (40, 80)]:
        lo, hi = wilson(k, n)
        assert lo <= k / n <= hi


def test_interval_narrows_as_n_grows() -> None:
    widths = [(lambda t: t[1] - t[0])(wilson(n // 2, n)) for n in (10, 100, 1000)]
    assert widths == sorted(widths, reverse=True), f"not monotonically narrowing: {widths}"


def test_no_observations_is_total_uncertainty() -> None:
    assert wilson(0, 0) == (0.0, 1.0)


def test_matches_known_values() -> None:
    """Hand-checkable reference points, so a transcription error in the
    arithmetic cannot pass the property tests above."""
    lo, hi = wilson(10, 20)
    assert round(lo, 4) == 0.2993
    assert round(hi, 4) == 0.7007

    # The sweep's headline row: 6 of 6 hits wrong at threshold 0.95.
    lo, hi = wilson(6, 6)
    assert round(lo, 2) == 0.61
    assert hi == 1.0
