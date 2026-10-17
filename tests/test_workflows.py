import json

from glm_review.red_team import run_red_team
from glm_review.review import run_reviews
from glm_review.storage import read_jsonl
from glm_review.synthesis import run_synthesis


COMPLETE_REVIEW = """## Summary and Strengths
Clear contribution.
## Major Concern
Missing control.
## Evidence and Reasoning
The comparison does not isolate the mechanism.
## Required Revision
Add the control.
## Recommendation Impact
Major revision.
"""

COMPLETE_SYNTHESIS = """## TL;DR
Run the pilot first.
## Points of Convergence
- Samples 1 and 2 agree.
## Points of Disagreement or Single-Sample Claims
- None.
## Final Recommendation
Proceed only if the pilot passes.
"""


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.responses.pop(0)


def test_review_persists_and_resumes(tmp_path):
    paper = tmp_path / "paper.tex"
    paper.write_text("A paper draft")
    output = tmp_path / "reviews"
    client = FakeClient([[COMPLETE_REVIEW]])

    first = run_reviews(
        client,
        {"paper": str(paper)},
        output,
        persona_ids=["methods-skeptic"],
    )
    assert first == {"completed": 1, "skipped": 0, "failed": 0}
    assert "Missing control" in (output / "paper/methods-skeptic.md").read_text()

    second = run_reviews(
        client,
        {"paper": str(paper)},
        output,
        persona_ids=["methods-skeptic"],
    )
    assert second == {"completed": 0, "skipped": 1, "failed": 0}
    assert len(client.calls) == 1


def test_review_rejects_incomplete_output(tmp_path):
    paper = tmp_path / "paper.tex"
    paper.write_text("A paper draft")
    counts = run_reviews(
        FakeClient([["truncated"]]),
        {"paper": str(paper)},
        tmp_path / "reviews",
        persona_ids=["clarity-reviewer"],
    )
    assert counts["failed"] == 1
    assert not (tmp_path / "reviews/paper/clarity-reviewer.md").exists()


def test_red_team_writes_independent_samples(tmp_path):
    client = FakeClient([["risk one", "risk two"]])
    output = tmp_path / "red-team"
    counts = run_red_team(
        client,
        [{"id": "pilot", "text": "Should this experiment proceed?"}],
        output,
        samples=2,
    )
    assert counts["completed"] == 1
    records = read_jsonl(output / "pilot.jsonl")
    assert [record["sample"] for record in records] == [0, 1]
    assert client.calls[0][1]["n"] == 2


def test_synthesis_reads_samples_and_persists_result(tmp_path):
    source = tmp_path / "red-team"
    source.mkdir()
    (source / "pilot.jsonl").write_text(
        "\n".join(
            json.dumps({"question_id": "pilot", "sample": i, "text": text})
            for i, text in enumerate(["risk one", "risk two"])
        )
        + "\n"
    )
    output = tmp_path / "synthesis"
    counts = run_synthesis(
        FakeClient([[COMPLETE_SYNTHESIS]]),
        source,
        [{"id": "pilot", "text": "Should this experiment proceed?"}],
        output,
    )
    assert counts == {"completed": 1, "skipped": 0, "failed": 0}
    assert "Run the pilot first" in (output / "pilot.md").read_text()
