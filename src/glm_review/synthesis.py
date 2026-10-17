"""Synthesize independent red-team samples into an auditable recommendation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .client import GLMClient
from .storage import atomic_write, read_json, read_jsonl

HEADERS = (
    "## TL;DR",
    "## Points of Convergence",
    "## Points of Disagreement or Single-Sample Claims",
    "## Final Recommendation",
)


def is_complete_synthesis(text: str) -> bool:
    positions = [text.find(header) for header in HEADERS]
    return all(position >= 0 for position in positions) and positions == sorted(positions)


def build_synthesis_messages(
    question: str, samples: list[str], *, excerpt_chars: int = 3_000
) -> list[dict[str, str]]:
    sample_text = "\n\n".join(
        f"[Sample {index + 1}, concluding excerpt]\n...{text[-excerpt_chars:]}"
        for index, text in enumerate(samples)
    )
    prompt = f"""Question under review:
{question}

The following responses were generated independently; they did not see one another:

{sample_text}

Synthesize them using exactly these sections and no others:

## TL;DR
## Points of Convergence
## Points of Disagreement or Single-Sample Claims
## Final Recommendation

Cite sample numbers. Treat independent convergence as stronger evidence, but do not
turn repeated speculation into fact. Do not expose private chain-of-thought.
"""
    return [
        {
            "role": "system",
            "content": "Produce a concise, decision-oriented synthesis of independent critiques.",
        },
        {"role": "user", "content": prompt},
    ]


def run_synthesis(
    client: GLMClient,
    input_dir: Path,
    questions: list[dict[str, str]],
    output_dir: Path,
    *,
    overwrite: bool = False,
    max_tokens: int = 16_384,
) -> dict[str, int]:
    question_map = {item["id"]: item["text"] for item in questions}
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    for source in sorted(input_dir.glob("*.jsonl")):
        destination = output_dir / f"{source.stem}.md"
        if destination.exists() and not overwrite:
            counts["skipped"] += 1
            continue
        if source.stem not in question_map:
            print(f"[{source.stem}] missing question configuration")
            counts["failed"] += 1
            continue
        samples = [record["text"] for record in read_jsonl(source)]
        try:
            result = client.chat(
                build_synthesis_messages(question_map[source.stem], samples),
                temperature=0.2,
                max_tokens=max_tokens,
                reasoning_effort="max",
            )[0]
        except Exception as exc:
            print(f"[{source.stem}] request failed: {exc}")
            counts["failed"] += 1
            continue
        if not is_complete_synthesis(result):
            print(f"[{source.stem}] incomplete synthesis; route=retry")
            counts["failed"] += 1
            continue
        atomic_write(destination, f"# Red-Team Synthesis: {source.stem}\n\n{result.strip()}\n")
        counts["completed"] += 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wait", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    client = GLMClient()
    if args.wait:
        client.wait_until_ready()
    counts = run_synthesis(
        client,
        args.input,
        read_json(args.questions),
        args.output,
        overwrite=args.overwrite,
        max_tokens=args.max_tokens,
    )
    print(" ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
