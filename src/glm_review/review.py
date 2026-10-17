"""Resumable multi-persona review workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .client import GLMClient
from .personas import REVIEW_INSTRUCTIONS, is_complete_review, select_personas
from .storage import atomic_write, read_json


def build_review_messages(paper_text: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Review this paper draft:\n\n{paper_text}\n\n{REVIEW_INSTRUCTIONS}",
        },
    ]


def route_review(text: str) -> str:
    """Return the next workflow route for a generated review."""
    if not text.strip():
        return "retry-null"
    if not is_complete_review(text):
        return "retry-incomplete"
    return "persist"


def run_reviews(
    client: GLMClient,
    papers: dict[str, str],
    output_dir: Path,
    *,
    persona_ids: list[str] | None = None,
    overwrite: bool = False,
    max_tokens: int = 32_768,
) -> dict[str, int]:
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    for paper_id, paper_path in papers.items():
        paper_text = Path(paper_path).read_text(encoding="utf-8")
        for persona in select_personas(persona_ids):
            destination = output_dir / paper_id / f"{persona.identifier}.md"
            if destination.exists() and not overwrite:
                counts["skipped"] += 1
                continue
            messages = build_review_messages(paper_text, persona.system_prompt)
            try:
                review = client.chat(
                    messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    reasoning_effort=persona.reasoning_effort,
                )[0]
            except Exception as exc:
                print(f"[{paper_id}/{persona.identifier}] request failed: {exc}")
                counts["failed"] += 1
                continue
            route = route_review(review)
            if route != "persist":
                print(f"[{paper_id}/{persona.identifier}] {route}")
                counts["failed"] += 1
                continue
            content = f"# Review: {paper_id} — {persona.identifier}\n\n{review.strip()}\n"
            atomic_write(destination, content)
            counts["completed"] += 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", required=True, help="JSON object: paper ID to file path")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--persona", action="append", dest="personas")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32_768)
    parser.add_argument("--wait", action="store_true", help="wait for the model server")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    client = GLMClient()
    if args.wait:
        client.wait_until_ready()
    counts = run_reviews(
        client,
        read_json(args.papers),
        args.output,
        persona_ids=args.personas,
        overwrite=args.overwrite,
        max_tokens=args.max_tokens,
    )
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
