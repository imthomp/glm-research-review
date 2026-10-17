"""Reviewer roles and response-validation rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    identifier: str
    system_prompt: str
    reasoning_effort: str = "max"


PERSONAS = (
    Persona(
        "methods-skeptic",
        "Review experimental validity. Identify the strongest confound, missing "
        "control, or mismatch between the design and the claimed mechanism. Prefer "
        "one consequential issue over a padded list of minor concerns.",
    ),
    Persona(
        "stats-pedant",
        "Review statistical rigor: effect sizes, uncertainty, multiple comparisons, "
        "sample adequacy, and whether the prose is stronger than the evidence.",
    ),
    Persona(
        "so-what-reviewer",
        "Assume the reported results are correct and evaluate their importance, "
        "novelty, and practical or scientific consequence.",
    ),
    Persona(
        "related-work-completeness",
        "Identify the closest missing or under-engaged prior work. Mark uncertainty "
        "instead of inventing citations.",
    ),
    Persona(
        "reproducibility-reviewer",
        "Determine whether an independent researcher could reproduce the study. "
        "Flag missing data-processing decisions, hyperparameters, and procedures.",
    ),
    Persona(
        "devils-advocate",
        "Construct the single strongest alternative explanation for the headline "
        "finding and state what evidence would distinguish it.",
    ),
    Persona(
        "overclaiming-reviewer",
        "Compare abstract, introduction, and conclusion claims against the actual "
        "results and limitations. Identify claims that outrun the evidence.",
    ),
    Persona(
        "fair-but-thorough",
        "Give specific credit for strengths, then identify the two or three issues "
        "that would genuinely change an acceptance recommendation.",
    ),
    Persona(
        "numbers-auditor",
        "Audit numbers available in the text for arithmetic inconsistency, statistical "
        "implausibility, and prose-versus-table contradictions. Cite each number used.",
        reasoning_effort="high",
    ),
    Persona(
        "clarity-reviewer",
        "Review organization and exposition. Identify the smallest concrete edits "
        "that would make the contribution and evidence easier to understand.",
    ),
)

REVIEW_INSTRUCTIONS = """Return a self-contained review using exactly these sections:

## Summary and Strengths
## Major Concern
## Evidence and Reasoning
## Required Revision
## Recommendation Impact

Do not expose private chain-of-thought. Be concrete, distinguish evidence from
speculation, and do not invent citations or experimental results.
"""


def select_personas(identifiers: list[str] | None = None) -> tuple[Persona, ...]:
    if not identifiers:
        return PERSONAS
    requested = set(identifiers)
    selected = tuple(persona for persona in PERSONAS if persona.identifier in requested)
    missing = requested - {persona.identifier for persona in selected}
    if missing:
        raise ValueError(f"unknown personas: {', '.join(sorted(missing))}")
    return selected


def is_complete_review(text: str) -> bool:
    required = (
        "## Summary and Strengths",
        "## Major Concern",
        "## Evidence and Reasoning",
        "## Required Revision",
        "## Recommendation Impact",
    )
    positions = [text.find(header) for header in required]
    return all(position >= 0 for position in positions) and positions == sorted(positions)
