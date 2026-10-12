# Zils Python SDK

Python 3.11+. Includes the `zils` CLI; install the `mcp` extra for `zils-mcp`.
This package is distributed from source and release artifacts, not PyPI.
From a clone of `ooo-hq/zils-sdk`:

```sh
python -m pip install "./packages/python[mcp]"
zils login --api-key
zils models
```

Applications use `ZILS_API_KEY` or an explicit key, independently of CLI login:

```python
from zils_sdk import Noul, ZilsClient

with ZilsClient() as client:
    result = client.system_one(
        state="Please refund the duplicate charge.",
        questions={"refund": Noul(instructions="Is a refund requested?")},
    )
    print(result.nouls["refund"].noul)
```

`AsyncZilsClient` provides the asynchronous equivalent. Default endpoint:
`https://training.zils.ai/decision`; default model: `zils-shared`. Override with
`base_url`/`model` or `ZILS_BASE_URL`/`ZILS_DEFAULT_MODEL`. Automatic retries are
off; a repeated decision can incur another charge. The default timeout is 60s.

The CLI and MCP share `~/.zils/config.json`; `ZILS_CONFIG_PATH` overrides its path.
`ZILS_API_KEY` overrides saved credentials. MCP exposes `list_models` and `decide`.

## Rank a short list

Version 0.2.0 adds experimental `client.rank`:

```python
from zils_sdk import ZilsClient

items = ["Update the logo", "Checkout is failing", "Fix a typo"]
with ZilsClient() as client:
    result = client.rank(items, "Most urgent customer impact first")
    for entry in result.ranking:
        print(items[entry.index], entry.score)
    print(result.comparisons, result.usage.billable_input_tokens)
```

`AsyncZilsClient.rank` accepts the same arguments with `await`. Both accept 2–10
JSON items and the usual per-call `model`, `timeout`, and `retry` options.
Every pair is compared in both orders in one existing decision API request:
`n * (n - 1)` model judgments, up to 90. The SDK fits Bradley–Terry strengths
locally; 1 means strongest in this set, not certainty or absolute quality. Equal
scores retain input order. Results include original indices and unchanged usage.
Existing context and billing limits still apply. This source feature does not
add a new HTTP endpoint, CLI command, or MCP tool. See the
[ranking contract](https://github.com/ooo-hq/zils-sdk/blob/main/docs/developer-tools.md#ranking)
for numerical limits and failure behavior.

Evaluate ranking on your own task before relying on the order. Use ordinary code
for arithmetic, unit conversion, and purely numeric sorting; the model can make
incorrect comparisons, and fitting strengths does not correct that evidence.

[Complete guide](https://github.com/ooo-hq/zils-sdk/blob/main/docs/developer-tools.md).

For browser account sign-in use `zils login`. For H2O and JevK5 text training, see the
[training CLI guide](https://github.com/ooo-hq/zils-sdk/blob/main/docs/training-cli.md).
Browser login does not create an inference key.
