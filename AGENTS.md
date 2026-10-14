# Client toolkit standards

This repository owns the Python and TypeScript SDKs, CLI, and MCP server. Keep
applications independent of subnet and platform source. The HTTP contract is
the integration boundary. Training, billing administration, and serving belong
in their own repositories.

Preserve TypeSafe compatibility, separate billable/resource token counts, secure
credential handling, and zero automatic retries by default. Never put keys in
MCP tool arguments or CLI command arguments. Preserve upstream license notices.

Run `make check` for independent client/protocol tests. Use `make check-integration`
with a local platform checkout for the real gateway; never use live credentials
or spend credit for routine checks. The npm packages `@zils/sdk` and `@zils/cli`
are published; publish only through the separate release action, never from
routine checks. Python distributions ship on the access-controlled GitHub
release, not PyPI. Keep local plans, logs and machine paths in ignored `.private/`.
