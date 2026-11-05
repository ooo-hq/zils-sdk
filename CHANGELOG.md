# Changelog

## Unreleased

- The MCP server can train. New tools: `train_check` (local split checks and
  the most a job can cost), `train_submit`, `train_list`, `train_status` and
  `train_cancel`. They use the browser account login from `zils login`, never
  a key in tool arguments. `train_submit` requires `max_cost_usd` and refuses
  before sending anything when the job could cost more.
- `zils train submit --max-cost <USD>` does the same check in the terminal.
  The ceiling prices every train case at the 8,192-token cap:
  `max($2, cases × 8,192 × $0.20 / 1M)`.
- One-line agent setup: `zils mcp` runs the MCP server, and the npm CLI's
  private runtime now installs the MCP library (best effort, like `zils
  merge`). Add it with `claude mcp add zils -- npx -y @zils/cli mcp` or the
  Codex equivalent; no source checkout or separate Python install.

## 0.4.11 — 2026-11-01

- `zils train` explains running out of credit. When the service refuses to
  start a job for credit (HTTP 402), the message says so, links to
  https://zils.ai/billing and prints the exact `zils train resume <job> --train
  … --calibration … --test …` command: the job keeps its uploads, so after
  adding credit, resume submits it without uploading again. It used to say only
  "Training service returned HTTP 402. Check the job and account".
- `zils train status` and `list` show why a job failed, for example the price
  when a dataset costs more than the available credit. Cancelled jobs are shown
  as before. Training errors still never echo the service's error responses.
- The TypeScript SDK is versioned alongside the CLI; its runtime is unchanged.

## 0.4.10 — 2026-11-01

- `zils merge --pool d3-lite` (or another soup pool) checks the result as a
  soup body right after merging, with the Zils miner's own check
  (`zils-miner commit-body --check`). It prints why a folder is not a body yet,
  for example a merge of text-only exports, which has no vision tower. Without
  the miner it prints the install line and the check to run later; a miner too
  old for the check is told to update.
- `zils merge` accepts a zero portion, as long as one model's is above zero.
  `--weights 0,1` writes the second model's exact values in the first model's
  layout, keeping the first model's other tensors. That turns a text-only
  fine-tune into a full Qwen3.5 model with its parent's vision tower:
  `zils merge Qwen/Qwen3.5-0.8B <cousin> --weights 0,1 --pool d3-lite`.
- Merges keep signed zeros: a weight that is −0 in every model stays −0
  (it was written as +0).
- The TypeScript SDK is versioned alongside the CLI; its runtime is unchanged.

## 0.4.9 — 2026-11-01

- `zils` on its own prints the Zils wordmark and the help (it used to exit
  asking for a command); `zils --help` and `-h` show the wordmark too. Other
  commands stay quiet, and the wordmark is plain text when output is not a
  terminal or `NO_COLOR` is set.
- Includes 0.4.8 (miner commands run `zils-miner`), which was released on
  GitHub but not published to npm.
- The TypeScript SDK is versioned alongside the CLI; its runtime is unchanged.

## 0.4.8 — 2026-11-01

- One `zils` for customers and miners. The miner commands (`setup`, `doctor`,
  `update`, `commit-candidate`, `commit-body`, `announce-serving`, `serve`) now
  run the Zils miner's own command, `zils-miner`, with the same arguments and
  exit code. Without a miner install, `zils doctor` and the others say how to
  get one (`curl -fsSL https://zils.ai/install.sh | sh`). `zils --help` lists
  both groups.
- The miner installer used to link the miner as `zils`, which hid this CLI on
  machines with both. It now links `zils-miner` and removes that old link.
- The TypeScript SDK is versioned alongside the CLI; its runtime is unchanged.

## 0.4.7 — 2026-11-01

- `zils merge` cooks a soup: it merges 2–5 models of the same family. Every
  model gets an even portion by default, as in the subnet's soup;
  `--weights 2,1,1` sets portions in order. `--weight` (two models) and
  `--method slerp` (exactly two) work as before.
- `merge.json` records `weights` (one per model) instead of `weight`, and
  other models' licences are copied as `LICENSE-model-2`, `LICENSE-model-3`, ….
- The TypeScript SDK is versioned alongside the CLI; its runtime is unchanged.

## 0.4.6 — 2026-11-01

- `zils merge` works on Python 3.15. It downloads Hugging Face models with its
  own small downloader (httpx, already a dependency) instead of
  huggingface_hub, so its only extra dependency is numpy. Downloads resume
  where they stopped, weight files are checked against the SHA-256 the Hub
  lists, and `HF_TOKEN` (for gated or private models) is sent to Hugging Face
  only. Models are cached under `zils/models` in your cache folder.
- The TypeScript SDK is versioned alongside the CLI; its runtime is unchanged.

## 0.4.5 — 2026-11-01

- Fix: on Python 3.15 the npm CLI's first-run setup failed and no command
  worked, because a `zils merge` dependency (pyyaml, via huggingface_hub) has
  no 3.15 build yet. Setup now installs the CLI first and `zils merge`'s
  dependencies separately; if they can't install, every other command still
  works and `zils merge` says how to use Python 3.11–3.14.
- The TypeScript SDK is versioned alongside the CLI; its runtime is unchanged.

## 0.4.4 — 2026-11-01

- `zils merge FIRST SECOND --out DIR` merges two models of the same family
  (folders or Hugging Face repo ids) on your machine, by linear average or
  SLERP, keeping the first model's files and layout. It needs numpy and
  huggingface_hub: the npm CLI installs them; from source use
  `pip install 'zils-sdk[merge]'`.
- `zils merge` ignores the second model's extra output heads (`mtp.*`,
  `lm_head`) and vision tower when the first model has none, instead of
  refusing, so d3 models merge with Qwen bases and their fine-tunes.
- The TypeScript SDK is versioned alongside the CLI; its runtime is unchanged.

## 0.4.3 — 2026-10-23

- Training help and current documentation use the z3 public model-line names.
- Existing d3 pool keys, model profiles and API behavior remain unchanged.
- The TypeScript SDK is versioned alongside the CLI; its runtime is unchanged.

## 0.4.2 — 2026-10-21

- One sentence to paste into a coding agent: "Use `npx -y @zils/cli fit agent` to
  install the Zils skill, then see where Zils fits in this repo." The skill checks
  sign-in first and runs `zils login --yes`, so the person only approves the
  browser sign-in. `zils login --yes` creates the machine's API key without a
  terminal prompt. `zils fit agent` tells the agent to read the skill and continue.

- `zils fit agent` installs a `zils-fit` skill into the project (`.claude/skills/`
  for Claude Code, `.agents/skills/` for other agents). Then ask your coding agent
  "See where Zils fits in this repo": it finds the decisions, writes test cases
  from the app's own rules, runs them through `zils fit run`, applies the app's
  thresholds, and reports in plain words. It asks before spending over 100
  requests and doesn't change code unless asked.
- `zils fit run --answers FILE` saves every input with its expected answer and
  Zils's full answer, and the JSON report lists `agreed` and each disagreement
  (most confident first).
- API errors in the CLI include the service's own message.
- `zils fit` on its own (or `zils fit <folder>`) is now a guided first run. It
  opens with the Zils wordmark, then steps through reading the code, listing the
  questions your code asks an AI in plain words ("Is this cold email?", "Which
  status?"), trying one live on an example you paste, and how to check Zils on
  real traffic. It asks for the possible answers when the code builds them at run
  time. Piped output stays plain and asks nothing. `zils fit scan` and
  `zils fit run` are unchanged for scripts.

## 0.4.1 — 2026-10-21

- `zils login` now offers to create an API key for this machine after browser
  sign-in ("Create an API key for this machine? [Y/n]"), so `zils decide` and
  `zils fit run` work without a second login. The key is named after the host,
  saved owner-only, and can be revoked on zils.ai. Without a terminal, or with a
  key already saved, nothing is created.
- `zils fit scan` and `zils fit run` print grouped, aligned results with colour in
  a terminal (plain text when piped, with `NO_COLOR` or `TERM=dumb`; `--json` is
  unchanged). Free-text calls are counted and listed only with `--all`. `run`
  shows live progress.
- The scanner names arrow and generic functions, reads schemas defined above the
  call (`const schema = z.object(...)`), treats `z.enum(variable)` and
  classification fields such as `category: z.string()` as choices with TODO
  options, ignores providers mentioned in comments, lists one drop-in per file,
  and counts `.decide(` only with a questions body nearby. On Inbox Zero it now
  finds 6 decisions and 1 drop-in instead of 1 and 4.
- `zils fit run` stops after the first 5 requests fail and reports the error,
  instead of sending the whole log.

## 0.4.0 — 2026-10-21

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
  as it does for H2O. CLI training stays text-only.

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
