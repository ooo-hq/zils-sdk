# Where does Zils fit? (`zils fit`)

`zils fit` answers two questions about an existing codebase before you train
anything:

1. **Where could Zils go?** `zils fit scan` lists the model calls that make a typed
   decision (yes/no, pick one, a score) rather than write text.
2. **Would it work there?** `record` logs what those calls receive and answer in
   normal use; `zils fit run` replays the log on Zils and reports how much of that
   traffic Zils can take while agreeing with your current model.

## Start here

```sh
zils fit
```

Run it in your project folder (or pass a folder). It reads your code, lists the
questions your code asks an AI that Zils could answer, and lets you try one on an
example you paste. Trying needs `zils login`. The steps below are the same flow
for scripts and for measuring on real traffic.

## Let your coding agent do it

Paste this into your coding agent (Claude Code, Cursor, Codex) in your project:

> Use `npx -y @zils/cli fit agent` to install the Zils skill, then see where Zils fits in this repo.

Or install the skill yourself:

```sh
zils fit agent
```

This installs a `zils-fit` skill into the project for Claude Code
(`.claude/skills/`) and other agents (`.agents/skills/`). Then ask your agent:
"See where Zils fits in this repo." It reads the code, writes test cases from the
app's own definitions, runs them with `zils fit run`, and reports what Zils would
handle and where it disagrees. Its test cases are not real traffic; measuring on
real traffic is the steps below.

## 1. Scan

```sh
zils fit scan . --questions questions.json
```

Results are grouped. Free-text calls are only counted; add `--all` to list them,
or `--json` for machine-readable output. Each call shows what was found:

- `noul`, `choice`, `score`: a decision Zils can answer. The scan drafts a Zils
  question for it in `questions.json`.
- `drop-in`: the call already uses the Jev decision API shape. Point it at
  `https://training.zils.ai/decision` with a Zils model; no question to draft.
- `skip: free text`: the call writes text; Zils does not replace it.

The scan reads source text and makes no requests. It recognises common OpenAI,
Anthropic, Vercel AI SDK, LangChain and Jev-style calls in Python and
JavaScript/TypeScript, skips test files unless you pass `--include-tests`, and can
miss calls it does not recognise. Read `questions.json` and replace any `TODO`;
the wording is what Zils is asked.

## 2. Log

Wrap the decision function the scan pointed at. Its behaviour does not change.

```python
from zils_sdk import record

@record("is_urgent")
def is_urgent(ticket: str) -> bool:
    ...  # your existing model call
```

```ts
import { record } from '@zils/sdk';

export const routeTicket = record('department', async (ticket: string) => {
  // your existing model call
});
```

Use the question names from `questions.json`. Each call appends one line to
`fit-log.jsonl` (set `ZILS_FIT_LOG` to change it). The log contains your real
inputs: it is created readable by its owner only, and is never sent anywhere
until you run step 3. Failed calls are not logged, and a logging error never
reaches your code. A function with one argument logs that argument as the state.

A few hundred distinct inputs is a minimum; with fewer, thresholds move from run
to run.

## 3. Run

```sh
zils fit run --questions questions.json --log fit-log.jsonl
```

It needs an API key: `zils login` offers to create one after browser sign-in,
or use `zils login --api-key`.

This sends **one paid request per distinct logged input** to `zils-shared`
(stock d3-mini; `--model` picks another). `--max-rows` (default 500) caps the
spend. For each question it picks, on half of the inputs, the lowest confidence
at which Zils agrees with your logged answers at `--target` (default 95%), and
reports the other half:

- **handled**: share of inputs Zils would answer at that confidence; the rest go
  to your current model.
- **agreement when handled**: how often Zils matched your logged answer there.
- **status**: `ok`, `check_miss` (the threshold did not hold on the other half;
  add data), `no_threshold` (Zils never reached the target; consider training)
  or `too_few_rows` (the log is too small to show the target; at 95% that means
  fewer than about 150 inputs). `not_logged` means the log has no answers for it.

By default the threshold must clear the target at the 95% lower confidence bound;
`--point-estimate` uses the measured rate instead. Failed requests are listed and
left out, never counted as disagreements; if the first five all fail, the run
stops and shows the error.

**Agreement is not accuracy.** The logged answers are your current model's, so the
report measures whether Zils can replace that call without changing behaviour.
To measure who is right, put verified answers (real outcomes or human labels) in
the log's `answer` field.

If a question misses, the same logged inputs, with verified answers, are training
data for `zils train`.
