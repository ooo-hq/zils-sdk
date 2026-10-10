# Zils SDKs, CLI, and MCP

Python and TypeScript clients, the `zils` terminal command, and a stdio MCP
server for the Zils decision API, plus browser sign-in and text-training commands. Applications use HTTP; they do not need a
miner, validator, model weights, or the hosted backend's source code.

These packages are not published to PyPI or npm. Use the
[setup and usage guide](docs/developer-tools.md) to install from this repository.

```sh
python3 -m venv .venv-tools
. .venv-tools/bin/activate
python -m pip install "./packages/python[mcp]"
zils login --api-key
zils models
```

Use Python 3.11+ and Node.js 20+. Windows activation is
`.venv-tools\Scripts\Activate.ps1`. Login validates an existing Zils key and
stores it privately for CLI and MCP use. SDK applications supply their own key.
No Nace service account is needed. Automatic retries are off because a request
whose reply is lost may already have been billed.

Use **`zils login`** for browser account sign-in, then `zils train` to submit and
manage H2O and JevK5 text training. [Training CLI guide](docs/training-cli.md). Account
sessions and inference API keys are separate; browser login does not create a key.

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
