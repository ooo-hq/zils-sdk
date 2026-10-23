# Train from the Zils CLI

Install `@zils/cli` from npm using the
[developer-tools setup](developer-tools.md#install-from-npm). The CLI
defaults to the z3-mini founding baseline through its immutable d3-mini profile. Optional `--pool d3-lite`,
`--base-channel latest`, or `--model EXACT_PROFILE_ID` selects a qualified profile
advertised by the service. Service pool keys remain d3-lite, d3-nano, d3-mini, d3-flash and d3;
catalog membership alone does not mean a pool is available. H2O and JevK5 are retired for new
jobs; the CLI only downloads their older accepted results. Training execution remains in the hosted platform;
no GPU or model installation is needed on your laptop.

The public model line is **z3**. Existing API IDs, pool keys, downloaded artifacts
and pinned metadata keep their original `d3` identities. Pass the exact values
returned by discovery, including in `--pool` and `--model`; public z3 names are
not new API aliases. The founding mini baseline uses unchanged upstream weights.

## Sign in

```sh
zils login
```

Your browser opens Zils account sign-in and asks you to approve the CLI. Use your
existing invited account. Keep the terminal open and open email links on the same
computer. The fixed local callback uses port **43187**; a remote/headless shell is
not supported by this flow. Sign-in expires after ten minutes.

Browser sign-in requires the operator to enable the matching
[platform configuration](https://github.com/ooo-hq/zils-platform/blob/main/docs/cli-training.md)
and [website approval page](https://github.com/ooo-hq/zils-web/blob/main/docs/cli-sign-in.md).
The code is not evidence that a hosted service has enabled this beta OAuth flow.

The account session is saved privately to `~/.zils/config.account.json`, separate
from the inference key in `~/.zils/config.json`. Files use mode `0600` on POSIX;
on Windows, protect the directory with your account's filesystem permissions.
`ZILS_CONFIG_PATH` changes the key filename and derives the adjacent account file.
Tokens refresh under a process lock. An ambiguous refresh requires a fresh login.
Session writes are synchronized before refresh; a failed local state write stops
the exchange. POSIX filesystems must support directory synchronization. Windows
uses a write-through replacement; native Windows and power-loss testing remain
release checks.

For another deployment, set `ZILS_TRAINING_URL` **before signing in**. The default
is `https://training.zils.ai`. Changing that variable cannot send an existing
account token to a different service. `ZILS_BASE_URL` configures inference only.

Browser login does not issue an inference key. Use `zils login --api-key` to
privately enter an existing key, or `zils login --api-key -` to read from stdin.
`zils logout` removes both local credential types, retaining recovery receipts.
It does not revoke remote sessions/keys, delete jobs, or unset environment keys.

## Submit three reviewed splits

From the repository root, this synthetic example demonstrates the input format:

```sh
zils train submit --name ticket-routing \
  --train examples/training/train.jsonl \
  --calibration examples/training/calibration.jsonl \
  --test examples/training/test.jsonl \
  --min-accuracy 0.8 --min-brier-improvement 0.01 \
  --allow-training-data-export --json
```

This command creates a real job when connected to an enabled service. Run it only
when you intend to submit training. These tiny examples establish syntax, not
model quality; local tests do not spend credits or train on a live GPU.
The export flag explicitly permits training data to be copied to approved workers.
Both quality thresholds are required; acceptance may produce no qualifying model.
Before creation, the CLI prints the active training account, service, files,
selected immutable model, thresholds and export consent to stderr. Submission uses that account's training
allowance/credit under the service's rules; the CLI does not estimate a price.
JSON results remain on stdout.

Each UTF-8 JSONL case requires `id`, `group_id`, `family`, `state`, `question`, and
`label`. Labels are strings: `"false"`/`"true"` for `noul`, a criterion key for
`choice`, or a zero-based index string for `score`. Questions support at most
255 outcomes. The server enforces the pinned model limits (the z3-mini founding baseline allows 4,096 input tokens); the CLI never truncates data.

IDs must be globally unique. Related source groups and identical state/question
pairs must stay in one split. Calibration and test families must match and occur
in training. Limits are 128 MiB per split and 128 KiB per case. Image cases,
invalid JSON, duplicate keys, and non-finite numbers are rejected before creating
a job. Private snapshots ensure uploaded bytes match the validated file hashes.

Selection is checked against fresh discovery before one create request. When
combined, pool, channel and exact profile must agree. The exact profile is frozen
in the local receipt and checked on creation, upload and resume; channel changes
do not change an existing job. Without pool discovery, the mini stable default
continues to work. These flags require this source version of the CLI; a source
change does not publish an npm release or enable a hosted pool.

## Follow and recover a run

Copy the returned job UUID into `JOB_ID`:

```sh
zils train list --json
zils train status JOB_ID
zils train status JOB_ID --watch
```

Watch polls every five seconds until training completes or fails. Ctrl+C stops
watching without cancelling the job. An accepted result may still be activating;
a model ID appears only after API readiness is confirmed. A completed run can
also have `no_qualifying_model`, with no downloadable model.

If uploads stop, resume the **same** job with the unchanged source files:

```sh
zils train resume JOB_ID \
  --train examples/training/train.jsonl \
  --calibration examples/training/calibration.jsonl \
  --test examples/training/test.jsonl --json
```

Receipts under the private configuration directory's `training/` folder bind the
job to its account, service and exact input hashes. Resume requires that receipt;
it skips confirmed uploads and refreshes expired storage grants. It never creates
a replacement job. Successful submission removes owned dataset snapshots but
keeps the small receipt. Interrupted submissions retain snapshots privately.

If the create response is lost or fails, the CLI preserves a private `pending-*.json`
receipt without guessing a job ID. Run `zils train list` and inspect the website
before submitting again. A job discovered this way cannot automatically inherit
an unknown receipt; inspect or cancel that run explicitly. No mutation is retried
automatically. Private pending snapshots remain until you resolve the attempt.

## Download or cancel

```sh
zils train download JOB_ID --out ./accepted-model --json
zils train cancel JOB_ID
# Explicit confirmation for scripts:
zils train cancel JOB_ID --yes --json
```

Downloads require an accepted supported z3 result, or an older accepted H2O or JevK5 result. The CLI verifies provider/job/file
paths, release identity, model metadata and the backend-compatible checkpoint
SHA-256 before publishing four files to a **new** directory. Existing paths are
never overwritten. The filesystem must support hard links within the destination's
parent. Transfers are bounded to 512 MiB of checkpoint data plus a 1 MiB release
record. Files are saved privately. Verification does not execute model code.

Cancellation asks for confirmation in a terminal; scripts require `--yes`.
Already running workers may finish, and cancellation does not promise a refund
or undo costs already incurred. Use status to confirm the recorded outcome.

All commands accept `--json` and `--timeout SECONDS` (default 30). Download/upload
transfers also have a 120-second per-file ceiling. Results go to stdout; progress
and errors go to stderr. Exit codes are 0 for a completed CLI operation, 1 for a
service/transfer failure, 2 for invalid input/configuration, and 130 for Ctrl+C.
A completed CLI operation does not imply that training passed its quality targets.
