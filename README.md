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
manage d3-mini text training (H2O and JevK5 apply only to downloading older accepted results). [Training CLI guide](docs/training-cli.md). Account
sessions and inference API keys are separate; browser login does not create a key.

## Where does Zils fit?

Run `zils fit` in your project for a guided tour. `zils fit scan` finds the model calls in your code that make typed decisions,
`record` logs what they receive and answer, and `zils fit run` replays that log on
Zils to show how much of the traffic it can take. See [zils fit](docs/fit.md).

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
