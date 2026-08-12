"""The workload the cache measurement runs against.

Three strata, because a cache benchmark built only from repeated questions
measures nothing anyone doubted:

- `duplicate`  — same intent, reworded. A cache hit here is correct.
- `near_miss`  — one short token apart, different correct answer. A hit is WRONG.
- `unrelated`  — different tasks entirely. A hit is wrong.

The near-miss stratum is generated, not hand-picked, and that distinction is
load-bearing. Hand-picking pairs that happen to embed closely would prove only
that such pairs can be found. Generating them from a fixed grid of
task-templates crossed with one-token substitutions makes the result a property
of the axis rather than of the author's ingenuity, and it means the same grid
can be rerun against a different embedding model.

Every pair is deterministic. No seed is needed because nothing here samples;
the workload is the full cross product, so it reproduces exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Stratum = Literal["duplicate", "near_miss", "unrelated"]


@dataclass(frozen=True)
class Pair:
    """Two prompts and whether one answer may serve for both."""

    a: str
    b: str
    stratum: Stratum
    axis: str
    interchangeable: bool


# ── Near-miss axes ───────────────────────────────────────────────────────────
# Each axis is a way one short, low-information token flips the correct answer.
#
# Values are attached to their own template rather than crossed with every
# template on the axis. An earlier version did cross them and generated
# "Report the file size in seconds", which is not a near-miss pair, it is a
# nonsense pair, and two nonsense prompts embed close together for reasons that
# have nothing to do with the effect being measured. Keeping the grid honest
# costs verbosity here and buys a result that means what it says.
_NEAR_MISS: list[tuple[str, str, list[tuple[str, str]]]] = [
    # quantity
    (
        "quantity",
        "Summarize the article in {} bullet points.",
        [("3", "5"), ("2", "7"), ("4", "10")],
    ),
    ("quantity", "List {} reasons why this approach fails.", [("3", "5"), ("2", "7")]),
    ("quantity", "Give me {} examples of this pattern.", [("3", "5"), ("4", "10")]),
    ("quantity", "Write {} test cases for this function.", [("3", "8"), ("2", "6")]),
    ("quantity", "Name {} alternatives to this library.", [("3", "5"), ("2", "10")]),
    # format
    ("format", "Convert the following configuration to {}.", [("JSON", "YAML"), ("JSON", "TOML")]),
    ("format", "Return the parsed result as {}.", [("JSON", "XML"), ("JSON", "YAML")]),
    ("format", "Export this table as {}.", [("CSV", "XML"), ("CSV", "JSON")]),
    ("format", "Serialize the response body to {}.", [("JSON", "YAML")]),
    # direction
    ("direction", "Sort the results in {} order.", [("ascending", "descending")]),
    ("direction", "List the commits in {} order by date.", [("ascending", "descending")]),
    ("direction", "Rank these items by cost, {}.", [("increasing", "decreasing")]),
    # polarity
    (
        "polarity",
        "What are the {} of using a monorepo?",
        [("advantages", "disadvantages"), ("pros", "cons")],
    ),
    ("polarity", "Describe the {} of this migration strategy.", [("benefits", "drawbacks")]),
    ("polarity", "Summarize the {} of adopting this framework.", [("upsides", "downsides")]),
    # language
    (
        "language",
        "Translate the following paragraph into {}.",
        [("French", "German"), ("Spanish", "Japanese")],
    ),
    ("language", "Write the error message in {}.", [("French", "German")]),
    ("language", "Localize this UI string to {}.", [("Italian", "Portuguese")]),
    # unit — each template keeps only the units that make sense for it
    ("unit", "Convert the distance to {}.", [("miles", "kilometres")]),
    ("unit", "Report the file size in {}.", [("megabytes", "gigabytes")]),
    ("unit", "Express the duration in {}.", [("seconds", "minutes")]),
    ("unit", "Give the temperature in {}.", [("Celsius", "Fahrenheit")]),
    # boundary
    ("boundary", "Show the {} three entries in the log.", [("first", "last")]),
    ("boundary", "Return the {} page of results.", [("previous", "next")]),
    ("boundary", "Describe the {} step of the deployment.", [("first", "last")]),
    # negation
    ("negation", "Which of these numbers {} prime?", [("are", "are not")]),
    ("negation", "List the services that {} currently healthy.", [("are", "are not")]),
    ("negation", "Which dependencies {} pinned to a version?", [("are", "are not")]),
    ("negation", "Show the tests that {} passing.", [("are", "are never")]),
]

# ── Duplicate paraphrases ────────────────────────────────────────────────────
# Same request, different wording. A cache that misses these is not earning its
# complexity, so this stratum is the reason to have a cache at all.
_DUPLICATES: list[tuple[str, str]] = [
    (
        "Summarize this document in three bullets.",
        "Give me a three-bullet summary of this document.",
    ),
    ("What does this function do?", "Explain the behaviour of this function."),
    ("How do I install the package?", "What are the installation steps for the package?"),
    ("Convert this dictionary to JSON.", "Serialize this dictionary as JSON."),
    ("Find the bug in the code below.", "What is wrong with the code below?"),
    ("Write a unit test for this method.", "Add a unit test covering this method."),
    ("What is the time complexity here?", "How does this scale with input size?"),
    ("Rename the variable to something clearer.", "Suggest a clearer name for this variable."),
    ("Explain this error message.", "What does this error mean?"),
    ("List the required environment variables.", "Which environment variables does this need?"),
    ("Refactor this to be more readable.", "Clean up this code so it reads better."),
    ("Is this query using an index?", "Does this query hit an index?"),
    ("Summarize the meeting notes.", "Give me the key points from the meeting notes."),
    ("How do I roll back the deployment?", "What are the steps to revert the deployment?"),
    ("Document this API endpoint.", "Write documentation for this API endpoint."),
    (
        "What changed between these two versions?",
        "Describe the differences between these versions.",
    ),
    ("Translate this comment into English.", "Render this comment in English."),
    ("Why is the build failing?", "What is causing the build failure?"),
    (
        "Generate a regex for email addresses.",
        "Write a regular expression matching email addresses.",
    ),
    ("Reduce the memory usage of this script.", "Make this script use less memory."),
]

# ── Unrelated prompts ────────────────────────────────────────────────────────
# Drawn from separate domains. Any hit against these is a false hit and needs no
# judgement call to classify.
_UNRELATED: list[str] = [
    "What is the capital of Portugal?",
    "Write a haiku about winter rain.",
    "How do I proof sourdough overnight?",
    "Explain the offside rule in football.",
    "What year did the Apollo programme end?",
    "Recommend a beginner road bike.",
    "How does a heat pump work?",
    "Summarize the plot of Middlemarch.",
    "What is the boiling point of ethanol?",
    "How do I repot a fiddle-leaf fig?",
    "Explain compound interest to a child.",
    "What causes the northern lights?",
    "Draft a thank-you note for a job interview.",
    "How long should I rest between sets?",
    "What is the difference between jam and preserve?",
    "Describe the rules of shogi.",
    "When should I prune apple trees?",
    "What is a reasonable tip in Lisbon?",
    "How do noise-cancelling headphones work?",
    "Explain why the sky is blue.",
]


def near_miss_pairs() -> list[Pair]:
    return [
        Pair(
            a=template.format(va),
            b=template.format(vb),
            stratum="near_miss",
            axis=axis,
            interchangeable=False,
        )
        for axis, template, values in _NEAR_MISS
        for va, vb in values
    ]


def duplicate_pairs() -> list[Pair]:
    return [
        Pair(a=a, b=b, stratum="duplicate", axis="paraphrase", interchangeable=True)
        for a, b in _DUPLICATES
    ]


def unrelated_pairs() -> list[Pair]:
    """Consecutive prompts paired off, plus a wrap-around, so every prompt
    appears in exactly two pairs and the stratum is not dominated by one item."""
    n = len(_UNRELATED)
    return [
        Pair(
            a=_UNRELATED[i],
            b=_UNRELATED[(i + 1) % n],
            stratum="unrelated",
            axis="cross_domain",
            interchangeable=False,
        )
        for i in range(n)
    ]


def full_workload() -> list[Pair]:
    return near_miss_pairs() + duplicate_pairs() + unrelated_pairs()


if __name__ == "__main__":
    from collections import Counter

    w = full_workload()
    print(f"{len(w)} pairs")
    for stratum, count in sorted(Counter(p.stratum for p in w).items()):
        print(f"  {stratum:10s} {count:4d}")
    print("near-miss axes:")
    for axis, count in sorted(Counter(p.axis for p in w if p.stratum == "near_miss").items()):
        print(f"  {axis:10s} {count:4d}")
