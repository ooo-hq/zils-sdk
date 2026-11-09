---
name: zils-fit
description: Use when someone asks where Zils fits in a codebase, whether Zils could replace an LLM or Jev call, how well Zils would work on their app's decisions, or wants Zils evaluated without writing test data themselves.
---

# Finding where Zils fits

Zils answers typed questions (yes/no `noul`, pick-one `choice`, rating `score`) about a JSON or text state. It does not write text. The `zils` CLI is the interface; `zils <command> --help` documents every flag. `zils fit scan` and `--help` are local. Each `zils decide` call is one paid request, failed requests included. `zils fit run` sends one request per distinct state: log lines with the same state are merged into one request that asks all their questions together.

## Before you start

- Use the `zils` command the user gave you (for example `npx -y @zils/cli`) wherever this says `zils`. If none was given and `zils` is not on PATH, use `npx -y @zils/cli` (needs Node.js 20+ and Python 3.11+).
- Run `zils models`. If it says to log in, run `zils login --yes` with a timeout of at least 10 minutes and tell the user: "Approve the Zils sign-in in the browser tab that just opened." It saves an API key for this machine; never ask the user to paste a key into the chat.

## Workflow

1. **Find the decisions.** Run `zils fit scan . --json`. Then read the code yourself: the scan is a starting list, not ground truth.
   - Prefer questions the app already defines for a decision model (files named like `decision-model/`, `systemone`, `decide(`): their wording, criteria and state shape are what production sends.
   - For LLM calls, read the prompt and output schema around each call site.
   - Note where options come from at run time (a database, user rules) and pick realistic values.
2. **Write the questions** to `zils-fit/questions.json`: `{id: {type, instructions, criteria}}`. `instructions` must be a string. If the app sends an object, join its parts and tell the user (see Known gaps). If the app changes a question's options from call to call, write one question id per variant and merge them in the report.
3. **Write test inputs** to `zils-fit/fit-log.jsonl`, one line per input and question: `{"question": id, "state": ..., "answer": expected}`. Use the app's real state shape. When the app asks several questions about one state in one call, write one line per question with the identical state; they go out as one request, as in production.
   - Take expected answers from the app's own definitions (prompts, rule text, docs), never your own taste.
   - Mix clear cases with borderline ones the definitions mention.
   - Aim for 15–30 per decision. Say up front how many requests that is and get a yes before spending more than 100. Pass the agreed budget as `--max-rows`; the run refuses to send more.
4. **Run** `zils fit run --questions zils-fit/questions.json --log zils-fit/fit-log.jsonl --answers zils-fit/answers.jsonl --json`. The report gives `agreed`, `agreement_all` and `disagreements` per question, scoring yes/no at 0.5. With under ~150 inputs, thresholds report `too_few_rows`; that is expected, so judge by agreement and confidence.
5. **Apply the app's own rules** when the code has them (minimum confidence before acting, cut-offs for yes/no, fallback when unsure, how several answers combine). Compute them from `answers.jsonl`: one line per input with `state`, `logged` (your expected answers by question id) and `zils` (Zils's full answers by question id: `noul` probability, or `probabilities` and `choice`), and report what Zils would decide versus hand back. Group related question ids back into the decision they belong to.

6. **Train if it misses.** When a question misses the target and the log holds right answers (real traffic logged with `record()`, or verified outcomes; never the test inputs you wrote in step 3), offer to train it. Agree the pass bars and open or private training with the user first (the `train_guide` MCP tool explains both), then run `zils fit train --log <log or folder> --questions zils-fit/questions.json --decide <id> --min-accuracy <agreed> --min-brier-improvement <agreed> --tier <open|private>` with `--allow-training-data-export` (and `--allow-open-tier-training` for open, `--max-cost` for private). For games or sessions, log one file per episode and pass the folder. When `zils train status <job id>` shows the API model, run `zils fit run --model <API model id>` on the same log and report before and after.

## Report shape

Write for someone who may not code. In order:

1. One sentence: where Zils fits and whether it is ready to try.
2. A table, one row per decision in plain words: cases, agreed, wrong, handed back.
3. Each disagreement in one line: the input gist, expected, Zils, confidence. Say if the expected label is arguable.
4. How to switch: the exact setting or code change, plus anything blocking it.
5. Caveats: inputs and labels were written by you, not real traffic; no comparison with the current model unless you ran one.
6. Next step: measuring on real traffic with `record()` (Python `from zils_sdk import record`, TypeScript `import { record } from '@zils/sdk'`) and `zils fit run`, then `zils fit train` if a question misses.

Do not change the app's code unless the user asks; offer the change instead.

## Switching to Zils

Zils speaks the Jev (TypeSafe System One) API: `POST https://training.zils.ai/decision/v1/systemone` with `Authorization: Bearer <key>` and the same `{model, state, questions}` body; answers come back in the same shape. The model for stock testing is `zils-shared`. So:

- Code already calling Jev or a System One endpoint: change the base URL, key and model.
- An app with a provider setting (for example `provider:model`): add a `zils` provider that reuses its System One client with the URL above and a `ZILS_API_KEY` setting, then select `zils:zils-shared`. Offer this change; make it only if asked.
- LLM calls: replace the call with a Zils request through the SDK (`ZilsClient().system_one(...)` in Python, `new ZilsClient().systemOne(...)` in TypeScript), keeping the LLM as the fallback below the app's confidence bar.

## Known gaps

- Zils requires `instructions` to be a string. Object-form instructions, which Jev accepts, return HTTP 422, and an app with an LLM fallback will silently fall back on those decisions.
- A 422 names no field. Check that `instructions` is a string, the `criteria` shape matches the type, and the state is non-empty.
