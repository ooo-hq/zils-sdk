# Changelog

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
