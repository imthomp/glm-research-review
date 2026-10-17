"""Generate independent critiques of planned research decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

from .client import GLMClient
from .storage import read_json, write_jsonl

SYSTEM_PROMPT = (
    "Red-team a research decision before resources are committed. Be concrete and "
    "falsifiable, focus on the cheapest consequential failure mode, and do not invent "
    "a problem merely to appear thorough. Do not expose private chain-of-thought."
)


def build_red_team_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def run_red_team(
    client: GLMClient,
    questions: list[dict[str, str]],
    output_dir: Path,
    *,
    samples: int = 8,
    overwrite: bool = False,
    max_tokens: int = 16_384,
) -> dict[str, int]:
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    for question in questions:
        destination = output_dir / f"{question['id']}.jsonl"
        if destination.exists() and not overwrite:
            counts["skipped"] += 1
            continue
        try:
            responses = client.chat(
                build_red_team_messages(question["text"]),
                n=samples,
                temperature=0.8,
                max_tokens=max_tokens,
                reasoning_effort="max",
            )
        except Exception as exc:
            print(f"[{question['id']}] request failed: {exc}")
            counts["failed"] += 1
            continue
        records = [
            {"question_id": question["id"], "sample": index, "text": text}
            for index, text in enumerate(responses)
            if text.strip()
        ]
        if len(records) != samples:
            print(f"[{question['id']}] incomplete: {len(records)}/{samples} non-null samples")
            counts["failed"] += 1
            continue
        write_jsonl(destination, records)
        counts["completed"] += 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wait", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    client = GLMClient()
    if args.wait:
        client.wait_until_ready()
    counts = run_red_team(
        client,
        read_json(args.questions),
        args.output,
        samples=args.samples,
        overwrite=args.overwrite,
        max_tokens=args.max_tokens,
    )
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
