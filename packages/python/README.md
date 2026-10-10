# Zils Python SDK

Python 3.11+. Includes the `zils` CLI; install the `mcp` extra for `zils-mcp`.
This package is not yet published. From a clone of `ooo-hq/zils-sdk`:

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

[Complete guide](https://github.com/ooo-hq/zils-sdk/blob/main/docs/developer-tools.md).

For browser account sign-in use `zils login`. For JevK5 text training, see the
[training CLI guide](https://github.com/ooo-hq/zils-sdk/blob/main/docs/training-cli.md).
Browser login does not create an inference key.
