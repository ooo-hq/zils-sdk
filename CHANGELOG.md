# Changelog

## Unreleased

- Add `zils fit`. `zils fit scan` lists model calls that make typed decisions
  (OpenAI, Anthropic, Vercel AI SDK, LangChain; Jev-shaped calls as drop-ins),
  skips test files, and drafts Zils questions. `record` (Python decorator and
  TypeScript wrapper) shadow-logs a decision function's inputs and answers to an
  owner-only JSONL file without changing its behaviour. `zils fit run` replays the
  log on Zils (one paid request per input, capped by `--max-rows`) and reports,
  per question, the share Zils handles and its agreement with the logged answers,
  with thresholds picked on half the inputs and checked on the other half.
  Agreement with your current model is not accuracy; see `docs/fit.md`.
- `zils train submit` accepts the d3-mini text profile the hosted service now
  offers (`d3-mini-4.5b-v1`, base `zils-ai/d3-mini-4.5b-v1` at revision
  `b54402a963373cc44713f18c9596c580e31cc68b`). Before this, the CLI refused every
  new job because the service no longer offers H2O or JevK5.
- New submissions require d3-mini. `zils train download` still accepts older
  H2O and JevK5 results, and requires the pinned profile in d3-mini `model.json`
  as it does for H2O. CLI training stays text-only; no version is bumped.

## 0.3.0 — 2026-10-13

- Add optional `group_size` / `groupSize` to Python sync/async and TypeScript
  `rank`. Groups accept 2–10 items, capped to the list length; four is a starting
  point for evaluation. Overlapping cyclic groups balance presentation positions
  and use `n` Choice judgments in one request, with a local Luce choice-strength
  fit across groups. Ten items use 10 judgments instead of 90.
- Preserve all-pairs ranking by default, the 2–10-item cap, result shape, input
  validation, and billing/retry behavior. Group probabilities are floored and
  renormalized before fitting; incomplete or invalid answers fail without repair
  requests. Shared synthetic fixtures cover strengths, position bias, short
  lists and duplicates. A small synthetic live pilot found lower ordering
  quality with grouped comparisons. Customer accuracy, public API latency and
  billed savings have not been measured; grouped ranking remains experimental.
- Update the npm CLI's bundled Python package to 0.3.0. No ranking CLI command,
  MCP tool, HTTP endpoint or model change is included.

## 0.2.0 — 2026-10-12

- Add experimental `rank` to the Python synchronous/asynchronous clients and the
  TypeScript client. Rank 2–10 JSON items against a criterion using the existing
  model, with both orders of every pair in one decision request.
- Return original item indices, relative Bradley–Terry strengths, comparison
  count, model identity, and unmodified token usage. No new model or HTTP route.
- Validate inputs before sending and reject incomplete or invalid comparisons.
  Existing timeout, cancellation, billing, and retry behavior is preserved.
- Update the npm CLI's bundled Python package to 0.2.0. No ranking CLI command
  or MCP tool is added.

Scores are not confidence or absolute quality. Ranking remains experimental:
evaluate it on the intended task, and use ordinary code for exact arithmetic and
numeric sorting. Ten items require 90 model judgments; a single HTTP request is
not a single model evaluation. See the [ranking guide](docs/developer-tools.md#ranking).
