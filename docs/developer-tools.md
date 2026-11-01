# Zils SDKs, CLI, and MCP

Use the same Zils decision API from Python, JavaScript/TypeScript, a terminal, or
an MCP-compatible agent. This first version lists available models and answers
yes/no (`noul`), choice, and ordered-score questions. The CLI also supports [d3-mini text training](training-cli.md) with browser
account sign-in. Image training, bulk jobs, miner/validator administration, and
billing management remain in their existing APIs.

The CLI and JavaScript/TypeScript SDK are distributed as `@zils/cli` and
`@zils/sdk` on npm. The Python SDK/MCP package is installed from source and is
not published on PyPI. No model download, GPU, or Nace service account is required.

## Install from npm

The CLI requires Node.js 20+ and Python 3.11+ with `venv` and `pip`:

```sh
npm install -g @zils/cli
zils login
zils train list
```

You can also use `npx @zils/cli login`. First use downloads the bundled Python
CLI's dependencies from PyPI into a private cache; subsequent starts reuse it.
No global Python packages are modified. Set `ZILS_PYTHON` to a Python executable
path if it is not on your PATH. See the [CLI package guide](../packages/cli/README.md).

For JavaScript/TypeScript applications, run `npm install @zils/sdk`.

## Install from a fresh clone

Use Python 3.11+ and Node.js 20+. The commands below use a macOS/Linux shell;
on Windows, activate the environment with `.venv-tools\Scripts\Activate.ps1`.

```sh
git clone https://github.com/ooo-hq/zils-sdk.git
cd zils-sdk
python3 -m venv .venv-tools
. .venv-tools/bin/activate
python -m pip install "./packages/python[mcp]"
zils --version
```

The Python distribution installs both `zils` and `zils-mcp`. Omit `[mcp]` if you
only need the Python SDK and CLI. For the JavaScript/TypeScript package:

```sh
npm ci --prefix packages/typescript
npm run build --prefix packages/typescript
npm pack ./packages/typescript
```

This writes `zils-sdk-0.4.5.tgz` in the current directory. Install that tarball
into your Node application with `npm install /path/to/zils-sdk-0.4.5.tgz`.
The installed import is `@zils/sdk`; both `import` and `require` are supported.

## Credentials and defaults

Bare `zils login` opens your browser to sign in with your Zils account for training.
It does not create an inference API key. `zils login --api-key` explicitly selects
the private key prompt described below. See [training sign-in](training-cli.md).

Create a customer API key in the Zils dashboard or through the
[key-management API](https://github.com/ooo-hq/zils-platform/blob/main/docs/decision-api.md#create-a-key-and-call-the-model).
`zils login --api-key` validates an existing key and stores it locally; it does not create
an account or issue a key.

```sh
zils login --api-key
zils models
```

Login prompts without displaying the key. It checks `GET /v1/models` before
atomically saving `~/.zils/config.json`, mode `0600` on POSIX. On Windows, protect
the directory with your account's filesystem permissions. Failed validation
leaves previous credentials intact. CLI and MCP read the same file. `zils logout`
removes both the saved key and browser account session; it does not revoke the key or unset environment variables.

| Setting | Purpose and default |
| --- | --- |
| `ZILS_API_KEY` | API key; required for SDK calls unless supplied explicitly |
| `ZILS_BASE_URL` | API base URL; `https://training.zils.ai/decision` |
| `ZILS_DEFAULT_MODEL` | Default model; `zils-shared` |
| `ZILS_CONFIG_PATH` | CLI/MCP credential file; `~/.zils/config.json` |

SDKs use explicit options, then the corresponding Zils environment variables,
then defaults. They do not read saved CLI credentials or TypeSafe credentials.
CLI/MCP use environment credentials when `ZILS_API_KEY` is set; otherwise they
read the saved key and URL, with `ZILS_BASE_URL` overriding the saved URL.
An environment key does not inherit a previously saved host. Keep keys on the
server; the TypeScript client rejects browser use.

Use the base URL **without** `/v1/systemone`. Remote endpoints require HTTPS;
HTTP loopback URLs are accepted for local development. Redirects are not followed.
For another Zils deployment, use `zils login --api-key --base-url https://YOUR_HOST/decision`.
For noninteractive login, pipe a key from your secret manager into
`zils login --api-key -`; do not place the key directly in command arguments.

## CLI

```sh
zils decide --state "Please refund the duplicate charge." \
  --questions '{"refund":{"type":"noul","instructions":"Does the customer request a refund?"}}'

zils decide --state @ticket.txt --questions @questions.json --json > answer.json
cat ticket.txt | zils decide --state - --questions @questions.json --json
zils models --json
```

`--state` accepts text or a JSON object/array. `--questions` accepts a JSON object.
`@file` reads a UTF-8 file; `-` reads stdin. Only one argument may read stdin.
Each input is bounded to 1 MiB; the server enforces its combined request limit.

For a complete request, save this as `request.json`:

```json
{
  "state": {"message": "Please refund the duplicate charge."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {"billing": "Payment problems", "support": "Product help"}
    }
  }
}
```

```sh
zils decide --body @request.json --json
```

Use `--model MODEL` to override the request/default model and `--timeout SECONDS`
for a per-request timeout. `--body` cannot be combined with `--state` or
`--questions`. Results go to stdout; status/errors go to stderr. Exit codes:
`0` success, `1` API/connection failure, `2` invalid input/configuration/file access,
`130` interrupted. Run `zils decide --help` for available flags.

## Python SDK

Set `ZILS_API_KEY` through your application's secret manager or pass `api_key`.

```python
from zils_sdk import Choice, Noul, ZilsClient

with ZilsClient() as client:
    result = client.system_one(
        state={"message": "Please refund the duplicate charge."},
        questions={
            "refund": Noul(instructions="Does the customer request a refund?"),
            "team": Choice(
                instructions="Which team?",
                criteria={"billing": "Payment problems", "support": "Product help"},
            ),
        },
    )
    print(result.nouls["refund"].noul)
    print(result.choices["team"].choice)
    print(result.usage.billable_input_tokens)
```

Use `AsyncZilsClient` with `async with` and `await` for asynchronous applications.
`client.models.list()` returns an object with `.models`. Questions can also be
raw dictionaries. `Score` criteria can contain structured JSON; answers preserve
their legend. `APIError.status` contains an HTTP status; `ZilsError` is the base
class for upstream SDK/transport errors. Invalid Zils configuration raises
`ValueError`. Python SDK timeouts are in seconds.

## TypeScript SDK

```ts
import { ZilsClient, choice, noul } from '@zils/sdk';

const client = new ZilsClient();
const result = await client.systemOne({
  state: { message: 'Please refund the duplicate charge.' },
  questions: {
    refund: noul('Does the customer request a refund?'),
    team: choice('Which team?', { billing: 'Payment problems', support: 'Product help' }),
  },
});
console.log(result.answers.refund.noul);
console.log(result.answers.team.choice); // inferred as 'billing' | 'support'
console.log(result.usage.billable_input_tokens);
```

`client.models.list()` returns an array of model cards. Constructor options
include `apiKey`, `baseURL`, `defaultModel`, and `timeout` in **milliseconds**.
Per-call options support `signal`, timeout, and retry overrides. Typed helpers,
answers, and error classes come from the compatible TypeSafe client.

## Ranking

SDK version 0.2.0 adds experimental `rank`, using the existing model and HTTP API.
Install TypeScript with `npm install @zils/sdk@^0.4.5`, or install the Python
checkout as described above.
Requests use the configured Zils endpoint and existing Choice decisions.

```ts
import { ZilsClient } from '@zils/sdk';

const client = new ZilsClient();
const items = ['Update the logo', 'Checkout is failing', 'Fix a typo'];
const result = await client.rank({
  items,
  instructions: 'Most urgent customer impact first',
});
for (const entry of result.ranking) console.log(items[entry.index], entry.score);
console.log(result.model, result.comparisons, result.usage.billable_input_tokens);
```

`rank(request, options)` returns a `Promise<RankResult>` and accepts the existing
per-call timeout, retry, and abort-signal options. An optional `request.model`
overrides the configured model. It does not expose `APIPromise.withResponse()`.

```python
from zils_sdk import ZilsClient

items = ["Update the logo", "Checkout is failing", "Fix a typo"]
with ZilsClient() as client:
    result = client.rank(items, "Most urgent customer impact first")
    for entry in result.ranking:
        print(items[entry.index], entry.score)
    print(result.model, result.comparisons, result.usage.billable_input_tokens)
```

Use `await client.rank(items, instructions)` with `AsyncZilsClient` for the async
equivalent. Python returns a `RankResult` with `RankedItem` entries; both types
are exported. Existing per-call model, timeout, retry, and extra-header options
are supported.

### Grouped ranking

Version 0.3.0 adds an optional group size. Pairwise ranking remains the default;
set the option explicitly to evaluate grouped ranking.

```ts
const result = await client.rank({
  items,
  instructions: 'Most urgent customer impact first',
  groupSize: 4,
});
```

```python
result = client.rank(items, "Most urgent customer impact first", group_size=4)
```

Python's async client accepts the same keyword. Omit the option (or use Python
`None`) for the existing all-pairs method. An explicit group size must be an
integer from 2 to 10; groups use `min(group_size, len(items))` distinct input
indices. The overall list still requires 2–10 items.

The SDK places items in input order on a cycle and compares one consecutive
group starting at each item, wrapping at the end. For five items and groups of
four, the questions cover `ABCD`, `BCDE`, `CDEA`, `DEAB`, and `EABC`. This connects
all items through overlapping groups and places every item in every position
once. A group size of 2 explicitly selects this sparse cycle; it does not select
the default all-pairs method. Groups as large as the list compare all items in
each of `n` rotated presentations.

| Items | Default all-pairs judgments | Grouped judgments (`groupSize: 4`) |
| --- | --- | --- |
| 2 | 2 | 2 |
| 3 | 6 | 3 |
| 10 | 90 | 10 |

Both methods use one HTTP request and return the same result shape. `comparisons`
counts Choice questions, so grouped mode returns `n`. Larger groups put more
text in each question and change the competing alternatives; they do not reduce
the number of grouped questions further. Context limits still apply. Fewer
questions do not guarantee proportional billing or latency savings.

Grouped mode fits a Luce choice model: within a group, an item's predicted
probability is its strength divided by the sum of strengths in that group.
The fit uses all returned probabilities as fractional outcomes, accounting for
the alternatives in each question. It does not compare raw probabilities across
different groups or treat one group as independent judgments of every pair.
Before fitting, each probability is floored at `0.001` and the group is
renormalized to sum to one. This changes extreme evidence to keep strengths
finite. The same convergence threshold, iteration limit and output rounding
described below apply.

Four items per group is a starting point for evaluation, not a validated quality
setting. A small synthetic live pilot found ordering regressions, including poor
fallback choices despite a correct first choice. Position balancing reduces a
fixed positional preference; it does not
remove content-dependent bias. Different input permutations change the groups
and can change rankings when judgments are inconsistent with one set of
strengths. Scores are relative to this list and method, not calibrated confidence.
Compare both methods on independently labeled examples before changing your
application's default; measure ranking quality, returned token usage and elapsed
time. The included synthetic tests establish arithmetic and protocol behavior,
not real-world ranking accuracy or savings.

### Contract and cost

- Supply 2–10 items and a nonempty instruction describing which items should
  rank highest. Items can be any finite JSON value, including structured records.
  Object keys must be strings; convert dates, sets, and other custom objects to
  JSON values yourself. Unsupported values are rejected without sending a request.
  Duplicate items remain separate entries, identified by their original indices.
- By default the SDK sends one `POST /v1/systemone` request containing `n * (n - 1)` Choice
  questions: every pair in both orders. Two items require 2 judgments, three
  require 6, and ten require 90. One request is not one model evaluation.
- The returned `ranking` is ordered strongest first. Each entry has a zero-based
  `index` into the original list and a `score` relative to the strongest fitted
  strength. Ties retain input order; tied leaders can all score 1.
- `comparisons` counts the Choice judgments: ordered pairs by default or groups when selected. `model`
  and `usage` come from the actual response. Billable and resource tokens remain
  distinct; a missing billable count stays missing. No cost is estimated locally.
- The existing body/context budgets, execution limits, billing, and timeout apply.
  Ten long documents can still exceed those limits. Automatic retries remain off
  by default; callers can explicitly opt in through the existing retry options.

### What the score means

The default pairwise method averages `P(A beats B)` with `1 - P(B beats A)` from the reversed
comparison. This reduces fixed presentation-order bias; it does not guarantee
that the judgments are unbiased or correct. It uses these probabilities as
fractional wins in a Bradley–Terry maximum-likelihood fit. The model's separate
Choice `confidence` is not used as an additional weight.

A fitted strength divided by the largest strength becomes the displayed score.
For example, illustrative scores of 1.0, 0.5, and 0.25 describe relative
strengths, not probabilities of being correct. Under the fitted model, a 0.5
item has a `0.5 / (0.5 + 1)`, or one-third, chance of beating the leader. Scores
do not sum to 1 and cannot be compared across unrelated lists or criteria.
The best item in a poor list can still score 1; there is no separate ranking
confidence or absolute quality threshold.

Exact arithmetic and unit conversion can produce incorrect model comparisons.
Compute derived numeric fields and normalize units in ordinary code first. If
the entire criterion is a numeric sort, use ordinary sorting directly. Comparing
both orders and fitting strengths does not correct wrong comparison evidence.

The [Bradley–Terry MM algorithm](https://doi.org/10.1214/aos/1079120141)
uses the complete comparison graph. Pair probabilities are clipped to
`[0.001, 0.999]` before fitting so zero/one judgments cannot force infinite
strength ratios. This bounds extreme evidence and changes those extreme inputs;
it is not a measured calibration of the model. The fit stops when the maximum
absolute log-strength update is below `1e-10`, with a 100,000-iteration ceiling.
Output strengths use ten significant digits, preserving tiny positive scores
while suppressing floating-point differences between ties.

Invalid inputs are rejected before the request. Missing comparisons, the wrong
answer type, invalid probabilities or confidence, missing/invalid model or token
usage, or a nonconvergent fit raise an error instead of returning a partial ranking.
A successful model request may
already be billed even if its response cannot be ranked. Transport/API failures
propagate normally; the SDK never silently substitutes per-item scoring or
submits a second request to repair an answer.

Tests cover known synthetic strengths, ties, cyclic preferences, fixed order
bias, extreme probabilities, structured/duplicate items, and the existing
gateway contract. They do not establish ranking accuracy, latency, or cost on
real customer data. Grouped fixtures also cover overlapping sets, balanced
positions, group-size validation, and both SDKs' agreement with known strengths.
There is no top-k shortcut, new model training, CLI command, or MCP tool.

## MCP

After installing the Python MCP extra, run `zils login --api-key` in the same user account
as your agent. Configure an MCP host to launch the installed `zils-mcp` executable
over stdio. Use the absolute path printed by `command -v zils-mcp` (Windows:
`where.exe zils-mcp`). A host configuration accepting `mcpServers` looks like:

```json
{
  "mcpServers": {
    "zils": {
      "command": "/absolute/path/to/.venv-tools/bin/zils-mcp"
    }
  }
}
```

Replace the example path. For Codex, register that executable with
`codex mcp add zils -- /absolute/path/to/.venv-tools/bin/zils-mcp`.
The MCP server exposes only `list_models` and `decide`. It starts without a key
so the agent can discover the tools; calls explain how to sign in. Login/key
changes take effect on the next tool call without restarting the server.
Credentials come from the environment or shared file, never from tool arguments.
Unknown tool arguments are rejected before sending a request.

## Usage and retries

Default request timeout is 60 seconds. Automatic retries are **off** across all
interfaces: an interrupted decision may already have completed and been billed.
SDK callers can explicitly opt into TypeSafe `RetryPolicy`/`retry` behavior when
appropriate. CLI and MCP do not retry requests automatically.

`usage.input_tokens` counts model resource use. `usage.billable_input_tokens`
counts logical input for billing and can be absent on older runtimes; never infer
it from the resource count. Python represents an absent count as `None` and
TypeScript as `undefined`. The response model can be the pinned version behind
the requested alias. Use an exact release ID when reproducibility requires it.
Model probabilities require evaluation for your task; installing a client does
not establish accuracy or serving capacity.

## Development checks and packaging

From the repository root in the activated environment:

```sh
python -m pip install -r requirements/dev.txt "./packages/python[mcp]" build
npm ci --prefix packages/typescript
make check PYTHON=python
python -m build packages/python
npm pack ./packages/typescript
```

Checks use a local HTTP contract fixture, CLI subprocesses, and the MCP stdio
protocol. Run `make check-integration PLATFORM_DIR=/path/to/zils-platform` in an
environment with the platform and subnet dependencies installed to run the same
checks against the real Zils gateway. They do not contact a hosted model
or spend credit. Python wheel/sdist files appear in `packages/python/dist/`.
Publishing either package is a separate release step.
