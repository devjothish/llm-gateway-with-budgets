"""Threshold sweep for the semantic cache.

Answers one question: is there a cosine threshold that delivers a useful hit
rate and a tolerable false-hit rate at the same time?

No LLM is called. The whole measurement is embedding similarity against a
workload whose ground truth is known by construction, which is what makes it
cheap enough to rerun on every embedding-model change.

    uv run python -m bench.sweep

Writes `results/cache-sweep-<model>.json` and prints the table.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from fastembed import TextEmbedding

from bench.stats import wilson
from bench.workload import Pair, full_workload

MODEL = "BAAI/bge-small-en-v1.5"
THRESHOLDS = [0.85, 0.90, 0.92, 0.95, 0.97, 0.99]


@dataclass(frozen=True)
class Scored:
    a: str
    b: str
    stratum: str
    axis: str
    interchangeable: bool
    cosine: float


@dataclass(frozen=True)
class Row:
    threshold: float
    hits: int
    pairs: int
    hit_rate: float
    hit_rate_ci: tuple[float, float]
    false_hits: int
    false_hit_rate: float
    false_hit_rate_ci: tuple[float, float]
    duplicates_caught: int
    duplicates_total: int


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v, strict=True))
    nu = sum(x * x for x in u) ** 0.5
    nv = sum(x * x for x in v) ** 0.5
    # float(): fastembed yields numpy float32, which json.dumps cannot encode.
    return float(dot / (nu * nv))


def score(pairs: list[Pair]) -> list[Scored]:
    """One embedding pass over the unique prompt set, then pairwise cosine."""
    embedder = TextEmbedding(model_name=MODEL)
    texts = sorted({p.a for p in pairs} | {p.b for p in pairs})
    vectors = dict(zip(texts, (list(v) for v in embedder.embed(texts)), strict=True))
    return [
        Scored(p.a, p.b, p.stratum, p.axis, p.interchangeable, _cosine(vectors[p.a], vectors[p.b]))
        for p in pairs
    ]


def sweep(scored: list[Scored]) -> list[Row]:
    duplicates = [s for s in scored if s.interchangeable]
    rows: list[Row] = []
    for t in THRESHOLDS:
        hits = [s for s in scored if s.cosine >= t]
        false_hits = [s for s in hits if not s.interchangeable]
        rows.append(
            Row(
                threshold=t,
                hits=len(hits),
                pairs=len(scored),
                hit_rate=len(hits) / len(scored),
                hit_rate_ci=wilson(len(hits), len(scored)),
                false_hits=len(false_hits),
                # Undefined with no hits. Reporting 0.0 there would read as
                # "perfectly safe" when the honest answer is "never fired".
                false_hit_rate=(len(false_hits) / len(hits)) if hits else float("nan"),
                false_hit_rate_ci=wilson(len(false_hits), len(hits))
                if hits
                else (float("nan"), float("nan")),
                duplicates_caught=sum(1 for s in duplicates if s.cosine >= t),
                duplicates_total=len(duplicates),
            )
        )
    return rows


def _distribution(scored: list[Scored], stratum: str) -> str:
    xs = sorted(s.cosine for s in scored if s.stratum == stratum)
    if not xs:
        return "-"
    return f"{xs[0]:.4f} / {statistics.median(xs):.4f} / {xs[-1]:.4f}"


def report(scored: list[Scored], rows: list[Row]) -> str:
    out: list[str] = []
    out.append(f"embedding model: {MODEL}   pairs: {len(scored)}\n")

    out.append("Cosine by stratum (min / median / max)")
    for stratum in ("duplicate", "near_miss", "unrelated"):
        n = sum(1 for s in scored if s.stratum == stratum)
        out.append(f"  {stratum:10s} n={n:<4d} {_distribution(scored, stratum)}")

    out.append("\nSeparation check")
    dups = [s.cosine for s in scored if s.stratum == "duplicate"]
    near = [s.cosine for s in scored if s.stratum == "near_miss"]
    if dups and near:
        overlap = sum(1 for x in near if x >= min(dups))
        out.append(f"  lowest duplicate      {min(dups):.4f}")
        out.append(f"  highest near-miss     {max(near):.4f}")
        out.append(
            f"  near-miss pairs at or above the lowest duplicate: {overlap}/{len(near)}"
            f"  ({overlap / len(near):.0%})"
        )
        if max(near) >= min(dups):
            out.append("  -> no threshold separates them. Any cut that keeps the")
            out.append("     duplicates also serves wrong answers.")

    out.append("\nThreshold sweep")
    out.append(f"  {'thresh':>7} {'hit rate':>20} {'false-hit rate':>22} {'dupes caught':>14}")
    for r in rows:
        lo, hi = r.false_hit_rate_ci
        hr = f"{r.hit_rate:.1%} [{r.hit_rate_ci[0]:.0%},{r.hit_rate_ci[1]:.0%}]"
        fh = f"{r.false_hit_rate:.1%} [{lo:.0%},{hi:.0%}]" if r.hits else "n/a (no hits)"
        out.append(
            f"  {r.threshold:>7.2f} {hr:>20} {fh:>22}"
            f" {r.duplicates_caught:>7}/{r.duplicates_total:<6}"
        )

    out.append("\nWorst near-miss pairs (highest cosine, all wrong to cache)")
    for s in sorted((s for s in scored if s.stratum == "near_miss"), key=lambda s: -s.cosine)[:5]:
        out.append(f"  {s.cosine:.4f}  [{s.axis}]")
        out.append(f"           A: {s.a}")
        out.append(f"           B: {s.b}")
    return "\n".join(out)


def main() -> None:
    pairs = full_workload()
    scored = score(pairs)
    rows = sweep(scored)

    text = report(scored, rows)
    print(text)

    out = Path("results")
    out.mkdir(exist_ok=True)
    path = out / f"cache-sweep-{MODEL.split('/')[-1]}.json"
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "thresholds": [asdict(r) for r in rows],
                "pairs": [asdict(s) for s in scored],
            },
            indent=2,
        )
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
