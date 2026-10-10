# Zils SDKs, CLI, and MCP

Use the same Zils decision API from Python, JavaScript/TypeScript, a terminal, or
an MCP-compatible agent. This first version lists available models and answers
yes/no (`noul`), choice, and ordered-score questions. The CLI also supports [JevK5 text training](training-cli.md) with browser
account sign-in. Image training, bulk jobs, miner/validator administration, and
billing management remain in their existing APIs.

These packages are built from this repository. They have **not been published to
PyPI or npm**. No model download, GPU, or Nace service account is required.

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

This writes `zils-sdk-0.1.0.tgz` in the current directory. Install that tarball
into your Node application with `npm install /path/to/zils-sdk-0.1.0.tgz`.
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
