"""Wilson score interval.

A deliberate copy of `horizon_bench.stats.wilson` from the sibling benchmark
project, not an import. The import was the original design and it was better
engineering inside the workspace, but this repo is published standalone, and a
path dependency on a directory that does not exist in the clone means
`uv sync` fails for anyone who tries to reproduce the measurement. Ten lines of
arithmetic is a smaller cost than a repo that will not build.

Kept honest with its own test in `tests/test_stats.py`, so this copy is
verified here rather than trusting that the original was.
"""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the Wald interval because Wald's coverage collapses exactly
    where these measurements live: proportions near 0 and 1 at modest n, where
    it cheerfully returns bounds outside [0, 1]. The 100%-false-hit row in the
    cache sweep is precisely that case.
    """
    if n <= 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = n + z2
    center = (k + z2 / 2) / denom
    half = z / denom * math.sqrt(k * (n - k) / n + z2 / 4)
    return (max(0.0, center - half), min(1.0, center + half))
