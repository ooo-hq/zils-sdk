# Zils

Small models for the decisions your app asks an LLM to make: yes or no, pick one
of a list, a score. Zils answers with a probability, and you can train your own
model on your data.

This repository holds the `zils` command, the Python and TypeScript clients, and
an MCP server for coding agents.

## Where does Zils fit in your code?

Paste this into Claude Code, Cursor or Codex in your project:

> Use `npx -y @zils/cli fit agent` to install the Zils skill, then see where Zils fits in this repo.

The agent finds the model calls in your code that make a decision rather than
write text, writes test cases from your app's own rules, asks Zils, and reports
what Zils would handle and where it disagrees. You only approve the Zils sign-in
in your browser.

To do it yourself, run `zils fit` in your project. `zils fit scan` lists the
calls, `record` logs what they receive and answer in normal use, and
`zils fit run` replays that log on Zils to show how much of the traffic it can
take. See [zils fit](docs/fit.md).

## Replace a classifier prompt

A typical LLM decision is a prompt and a string to parse:

```python
reply = openai.chat.completions.create(
    model="...",
    messages=[{"role": "user", "content": f"Does this customer want a refund? Answer yes or no.\n\n{ticket}"}],
)
wants_refund = reply.choices[0].message.content.strip().lower().startswith("yes")
```

With Zils the question is typed and the answer is a probability:

```python
from zils_sdk import Choice, Noul, ZilsClient

with ZilsClient() as client:
    result = client.system_one(
        state={"message": ticket},
        questions={
            "refund": Noul(instructions="Does the customer request a refund?"),
            "team": Choice(
                instructions="Which team?",
                criteria={"billing": "Payment problems", "support": "Product help"},
            ),
        },
    )
wants_refund = result.nouls["refund"].noul > 0.5  # probability of yes
team = result.choices["team"].choice
```

The same call in TypeScript:

```ts
import { ZilsClient, choice, noul } from '@zils/sdk';

const client = new ZilsClient();
const result = await client.systemOne({
  state: { message: ticket },
  questions: {
    refund: noul('Does the customer request a refund?'),
    team: choice('Which team?', { billing: 'Payment problems', support: 'Product help' }),
  },
});
```

Set `ZILS_API_KEY` on your server (create a key in the Zils dashboard). Questions
use `noul` (yes/no), `choice` and `score`. See the
[developer guide](docs/developer-tools.md) for every option.

## Train your own model

When the shared model is not accurate enough on your question, train one on your
labeled examples:

```sh
zils login        # browser sign-in
zils train guide  # what to decide before you train
zils train prepare tickets.csv --decide team --read message --group customer_id
zils train submit --name ticket-routing --train ... --calibration ... --test ... \
  --min-accuracy 0.8 --min-brier-improvement 0.01 --allow-training-data-export
zils train status JOB_ID --watch
```

`prepare` turns a CSV or JSON export into the three training files. Submitting
uses your account's training credit. See the [training guide](docs/training-cli.md).

## Merge models on your machine

`zils merge` blends 2–5 fine-tunes of the same base model into a new model
folder: a soup, like the one the Zils network builds every day. Each model is a
local folder or a Hugging Face repo id. Nothing is uploaded, hosted or scored,
and it needs no account:

```sh
zils merge vllm-sr/d3-lite eljiwo/jiwo-0.8b cribl-ai/cribl-decision-2-0.8B --out ./my-soup
zils merge ./model-a ./model-b ./model-c --weights 2,1,1 --out ./my-soup
zils merge ./model-a ./model-b --method slerp --weight 0.3 --out ./my-mix
```

- By default every model gets an even portion. `--weights` sets portions in
  order (normalised to sum to one); with two models, `--weight` is the second
  model's share.
- `--method slerp` follows the arc between exactly two models.
- The output keeps the first model's files and layout (config, tokenizer, any
  answer layer), and `merge.json` records the recipe. Every other model's
  licence is copied as `LICENSE-model-2`, `LICENSE-model-3`, …; check them
  before you share the result.
- Weight names line up across `model.` and `model.language_model.` prefixes,
  so text-only fine-tunes merge into a multimodal model. Weights not every
  model has are kept from the first. Other models' extra output heads
  (`mtp.*`, `lm_head`) and vision towers are ignored when the first model has
  none, so a d3 model merges with its Qwen base.
- Hugging Face models download into `zils/models` in your cache folder,
  resume if interrupted, and are checked against the Hub's SHA-256.
  Set `HF_TOKEN` for gated or private models.
- It refuses models whose shapes differ, weights the first model lacks, and
  repeated or identical models.

### Submit a merge to a network pool

To submit the result to a soup pool, add `--pool d3-lite` (or `d3-mini`): with
the Zils miner installed, the miner's body check runs on the new folder and says
whether it is a valid soup body, and why not. A body must be the full Qwen3.5
model with its vision tower, so list a full checkpoint (such as
`vllm-sr/d3-lite`) first; a merge of text-only exports alone needs the Qwen
parent's vision tower added before it can be a body. A mix of listed public
models is paid the scout rate, not the trainer rate.

To submit a text-only public fine-tune (such as a listed cousin) as a body, give
its Qwen parent a zero portion. The result has the cousin's exact language
weights and the parent's unchanged vision tower; edit the copied README to
credit the cousin before you upload it:

```sh
zils merge Qwen/Qwen3.5-0.8B cribl-ai/cribl-decision-2-0.8B --weights 0,1 --pool d3-lite --out ./cribl-body
```

## Rank a short list

`rank(...)` orders 2–10 JSON items against a criterion using Choice decisions,
comparing each pair in both orders and fitting relative strengths locally.
`group_size=4` (Python) or `groupSize: 4` (TypeScript) uses overlapping groups
instead: 10 judgments for 10 items rather than 90. Ranking is experimental:
scores are relative strengths, and accuracy needs task-specific evaluation. See
the [ranking guide](docs/developer-tools.md#ranking).

## Use from an agent (MCP)

```sh
claude mcp add zils -- npx -y @zils/cli mcp
```

The server is listed in the official MCP Registry as `ai.zils/zils`. See the
[CLI package guide](packages/cli/README.md) for other agents.

## Install

The CLI needs Node.js 20+ and Python 3.11+:

```sh
npm install -g @zils/cli    # or run any command with npx @zils/cli
npm install @zils/sdk       # TypeScript/JavaScript client
```

First use of the CLI sets up its bundled Python code in a private cache;
dependencies come from PyPI. The Python SDK installs from this repository:

```sh
python -m pip install "./packages/python[mcp]"
```

Automatic retries are off: a request whose reply is lost may already have been
billed. Browser sign-in (`zils login`) is for training; inference uses API keys,
and signing in does not create one.

## Mining

Miners use the same `zils` command: `zils setup`, `zils doctor`,
`zils commit-candidate` and the other miner commands run `zils-miner` from the
Zils miner install (`curl -fsSL https://zils.ai/install.sh | sh`, a separate
GPU-side install from the [Zils repository](https://github.com/ooo-hq/zils)).
Without it, those commands print the install line; `zils --help` lists both
groups.

## Checks

```sh
python -m pip install -r requirements/dev.txt "./packages/python[mcp]"
npm ci --prefix packages/typescript
make check PYTHON=python
```

The suite runs against local HTTP fixtures, CLI subprocesses, typed JavaScript
checks and MCP protocol tests. It sends no hosted requests and uses no credit.

## Licence

Each package carries its licence and a `THIRD_PARTY_NOTICES.md` for adapted
code.
