# Experiment history

This repository is a public-safe refactor of two internal experiment directories. Generated outputs and private paper drafts are not included.

## GLM-5.2 phase

- Served a local GLM-5.2 checkpoint behind vLLM's OpenAI-compatible API.
- Used eight parallel samples for self-consistency on research questions and pre-experiment red-teaming.
- Ran repeated reasoning rounds with periodic re-anchoring on the original question and a sliding context window.
- Added a separate synthesis pass that grouped independently convergent risks and isolated lower-confidence, single-sample claims.
- Added multi-persona review and cross-round comparison utilities for multilingual NLP manuscripts.

## GLM-5.3 phase

- Retargeted the workflows to GLM-5.3 without changing the external API contract.
- Expanded paper review to ten specialized personas across three manuscripts.
- Added explicit detection of null and structurally incomplete responses after long reasoning traces exhausted the output budget.
- Routed the numerical auditor to a smaller reasoning budget after repeated maximum-effort runs failed to emit a final answer.
- Used per-review artifacts and skip-on-exists behavior to resume preempted SLURM jobs without repeating completed inference.

## Public refactor

- Replaced user- and cluster-specific paths with JSON inputs, CLI flags, and environment variables.
- Centralized API calls, authentication, timeouts, and server readiness checks.
- Added atomic writes so preemption cannot leave an artifact that appears complete.
- Added offline unit and workflow tests plus CI for Python 3.10–3.12.
- Excluded model weights, logs, manuscripts, generated reviews, and node-discovery files.
