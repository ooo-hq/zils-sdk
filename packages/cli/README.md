# Zils CLI

Sign in with your Zils account, submit text training, follow runs, download
accepted models, and call the decision API.

Requires Node.js 20+ and **Python 3.11+**, including `venv` and `pip`. Install
Python from [python.org](https://www.python.org/downloads/). On Debian/Ubuntu,
install `python3-venv` if your Python installation does not include it.

```sh
npm install -g @zils/cli
zils login
zils train list
```

Or run without a global npm installation:

```sh
npx @zils/cli login
```

First use installs the bundled Python CLI and its dependencies from PyPI into a
private cache. It needs internet access, takes longer than subsequent starts,
and does not modify your system Python. npm installation itself runs no setup
scripts. Set `ZILS_PYTHON` to an executable path if Python is not on your PATH.

## Use from an agent (MCP)

```sh
claude mcp add zils -- npx -y @zils/cli mcp      # Claude Code
codex mcp add zils -- npx -y @zils/cli mcp       # Codex
```

Other hosts: run `npx` with the arguments `-y @zils/cli mcp` over stdio. The
agent can answer decisions and check, submit and follow training jobs, with a
required cost limit on every submission.

## Commands

```sh
zils --help
zils train submit --help
zils train list --json
zils train status JOB_ID --watch
zils train download JOB_ID --out ./accepted-model
zils train cancel JOB_ID
zils logout
```

`zils login` opens browser sign-in for training. Browser sign-in requires a local
desktop browser and port 43187. Training uses your account's allowance/credit;
only an accepted result can be downloaded. Login does not create an inference key.

For inference, supply an existing key at the private prompt:

```sh
zils login --api-key
zils models
zils decide --state "Please refund the duplicate charge." \
  --questions '{"refund":{"type":"noul","instructions":"Is a refund requested?"}}'
```

Keys are never supplied as command arguments. Existing `~/.zils` credentials and
training receipts are shared with the Python CLI. Automatic request retries are
off. `zils logout` removes local credentials; it does not revoke remote sessions.
Windows users should protect their account's cache and configuration directories.

## Submit training data

Prepare separate UTF-8 JSONL files for training, calibration, and test cases.
Every line is one case, for example:

```json
{"id":"train-1","group_id":"ticket-1","family":"refund","state":"Please refund the duplicate charge.","question":{"type":"noul","instructions":"Is a refund requested?"},"label":"true"}
```

Each case needs `id`, `group_id`, `family`, `state`, `question`, and `label`.
IDs must be unique across all splits. Related groups and identical state/question
pairs must stay in one split. Calibration and test families must match and occur
in training. Labels are strings: `"false"`/`"true"` for `noul`, a criterion key for
`choice`, or a zero-based index for `score`. Image cases are not supported.

```sh
zils train submit --name ticket-routing \
  --train train.jsonl --calibration calibration.jsonl --test test.jsonl \
  --min-accuracy 0.8 --min-brier-improvement 0.01 \
  --allow-training-data-export --json
```

This submits a real training job and permits data export to approved workers.
Both quality thresholds are required. Keep the returned job ID to follow the run.
Run `zils train submit --help` for the available model/profile options.
If an upload is interrupted, resume the same job with unchanged files:

```sh
zils train resume JOB_ID \
  --train train.jsonl --calibration calibration.jsonl --test test.jsonl
```

Do not submit again after an ambiguous connection failure: first inspect
`zils train list` and the website. Local recovery receipts are retained under
`~/.zils/training`. Cancellation does not undo incurred costs or promise a refund.

## Runtime cache

The runtime cache lives under `~/Library/Caches/zils/cli` on macOS,
`~/.cache/zils/cli` on Linux, or `%LOCALAPPDATA%\zils\cli` on Windows.
`XDG_CACHE_HOME`, when set, overrides the cache base. Removing the runtime cache
only forces setup on the next invocation; it does not remove credentials or jobs.

The CLI bundles the same Python implementation as `zils-sdk`; no model weights,
GPU runtime, or private repository access is required. The JavaScript/TypeScript
application client is available separately as `@zils/sdk`.
