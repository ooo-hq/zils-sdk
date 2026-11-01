# Zils SDKs, CLI, and MCP

Python and TypeScript clients, the `zils` terminal command, and a stdio MCP
server for the Zils decision API, plus browser sign-in and text-training commands. Applications use HTTP; they do not need a
miner, validator, model weights, or the hosted backend's source code.

Install the CLI from npm. It requires Node.js 20+ and Python 3.11+:

```sh
npm install -g @zils/cli
zils login
zils train list
```

Alternatively, run `npx @zils/cli login`. First use sets up the bundled Python
implementation in a private cache; dependencies are downloaded from PyPI.
Install the JavaScript/TypeScript application client with `npm install @zils/sdk`.
The Python SDK/MCP package remains available from source; it is not on PyPI.
See the [setup and usage guide](docs/developer-tools.md). No Nace service account
is needed. Automatic retries are off because a request whose reply is lost may
already have been billed.

Use **`zils login`** for browser account sign-in, then `zils train` to submit and
manage z3-mini text training (H2O and JevK5 apply only to downloading older accepted results). [Training CLI guide](docs/training-cli.md). Account
sessions and inference API keys are separate; browser login does not create a key.

Miners use the same `zils`: `zils setup`, `zils doctor`, `zils commit-candidate`
and the other miner commands run `zils-miner` from the Zils miner install
(`curl -fsSL https://zils.ai/install.sh | sh`, a separate GPU-side install from the
subnet repository). Without it, those commands print the install line; `zils --help`
lists both groups.

## Where does Zils fit?

Paste this into Claude Code, Cursor or Codex in your project:

> Use `npx -y @zils/cli fit agent` to install the Zils skill, then see where Zils fits in this repo.

The agent reads your code, writes test cases from your app's own rules, asks Zils,
and reports what Zils would handle. You only approve the Zils sign-in in your browser.

Run `zils fit` in your project for a guided tour. `zils fit scan` finds the model calls in your code that make typed decisions,
`record` logs what they receive and answer, and `zils fit run` replays that log on
Zils to show how much of the traffic it can take. See [zils fit](docs/fit.md).

## Merge models

`zils merge` blends 2–5 fine-tunes of the same base model into a new model
folder on your machine: a soup, like the one the subnet cooks every day. Each
model is a local folder or a Hugging Face repo id:

```sh
zils merge vllm-sr/d3-lite eljiwo/jiwo-0.8b cribl-ai/cribl-decision-2-0.8B --out ./my-soup
zils merge ./model-a ./model-b ./model-c --weights 2,1,1 --out ./my-soup
zils merge ./model-a ./model-b --method slerp --weight 0.3 --out ./my-mix
```

- By default every model gets an even portion, as in the subnet's soup.
  `--weights` sets portions in order (normalised to sum to one); with two
  models, `--weight` is the second model's share.
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
- To submit the result to a subnet soup pool, add `--pool d3-lite` (or
  `d3-mini`): with the Zils miner installed, the miner's body check runs on the
  new folder and says whether it is a valid soup body, and why not. A body must
  be the full Qwen3.5 model with its vision tower, so list a full checkpoint
  (such as `vllm-sr/d3-lite`) first; a merge of text-only exports alone needs
  the Qwen parent's vision tower added before it can be a body. A mix of listed
  public models is paid the scout rate, not the trainer rate.
- Hugging Face models download into `zils/models` in your cache folder,
  resume if interrupted, and are checked against the Hub's SHA-256.
  Set `HF_TOKEN` for gated or private models.
- It refuses models whose shapes differ, weights the first model lacks, and
  repeated or identical models. Nothing is uploaded, hosted or scored.

## Ranking

SDK version 0.2.0 adds `rank(...)` for ordering 2–10 JSON items against a
criterion. It uses existing Choice decisions, compares each pair in both orders,
and fits relative strengths locally. See the [ranking guide](docs/developer-tools.md#ranking)
for Python/TypeScript examples, usage accounting, and limits. Ranking is
experimental: scores are relative strengths, and accuracy needs task-specific evaluation.

Version 0.3.0 adds **optional grouped ranking**:
`groupSize: 4` in TypeScript or `group_size=4` in Python. Overlapping groups use
10 Choice judgments for 10 items; the default still uses all 90 ordered pairs.
See [grouped ranking](docs/developer-tools.md#grouped-ranking) for the
tradeoffs. Pairwise ranking remains the default.

## Checks

```sh
python -m pip install -r requirements/dev.txt "./packages/python[mcp]"
npm ci --prefix packages/typescript
make check PYTHON=python
```

The normal suite is independent: local HTTP fixtures, CLI subprocesses, typed
JavaScript checks, and MCP protocol tests. No hosted requests or credits are used.
For the real gateway check, install the subnet and platform source dependencies
following the platform README, then use that environment:

```sh
make check-integration PYTHON=/path/to/platform-environment/bin/python \
  PLATFORM_DIR=/path/to/zils-platform
```

That environment also needs this Python package's MCP extra. The platform
checkout supplies its gateway fixtures; the same client tests run against them.
This optional check does not require production credentials or database services.

## Related repositories

[Zils](https://github.com/ooo-hq/zils) owns miners and validators.
[zils-platform](https://github.com/ooo-hq/zils-platform) owns the hosted APIs,
billing, storage and orchestration. [zils-web](https://github.com/ooo-hq/zils-web)
owns the website. Some repositories may require organization access.

The initial client implementation was developed against Zils revision
`56d16880a69635a41ce6ecc26d0b0914056db88a`. It uses the compatible TypeSafe
clients and retains the required Nace notice for the adapted strict MCP argument
check. Package publication is a separate release step.
