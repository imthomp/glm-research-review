from glm_review.personas import is_complete_review, select_personas
from glm_review.review import route_review
from glm_review.synthesis import is_complete_synthesis


COMPLETE_REVIEW = """## Summary and Strengths
x
## Major Concern
x
## Evidence and Reasoning
x
## Required Revision
x
## Recommendation Impact
x
"""


def test_review_routes():
    assert route_review("") == "retry-null"
    assert route_review("partial") == "retry-incomplete"
    assert route_review(COMPLETE_REVIEW) == "persist"
    assert is_complete_review(COMPLETE_REVIEW)


def test_review_headers_must_be_ordered():
    reversed_review = "\n".join(reversed(COMPLETE_REVIEW.splitlines()))
    assert not is_complete_review(reversed_review)


def test_persona_selection_and_validation():
    selected = select_personas(["methods-skeptic", "numbers-auditor"])
    assert [persona.identifier for persona in selected] == [
        "methods-skeptic",
        "numbers-auditor",
    ]


def test_complete_synthesis():
    text = """## TL;DR
x
## Points of Convergence
x
## Points of Disagreement or Single-Sample Claims
x
## Final Recommendation
x
"""
    assert is_complete_synthesis(text)
