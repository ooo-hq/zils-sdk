# Zils TypeScript SDK

Node.js 20+. Typed decisions using the existing Zils API and TypeSafe-compatible
questions. This package is not yet published. From a clone of `ooo-hq/zils`:

```sh
npm ci --prefix packages/typescript
npm pack ./packages/typescript
```

Install the resulting `zils-sdk-0.1.0.tgz` into your application. Set the server's
`ZILS_API_KEY`, then use either ESM `import` or CommonJS `require`:

```ts
import { ZilsClient, choice } from '@zils/sdk';

const client = new ZilsClient();
const result = await client.systemOne({
  state: 'Please refund the duplicate charge.',
  questions: {
    team: choice('Which team?', { billing: 'Payments', support: 'Product help' }),
  },
});
console.log(result.answers.team.choice);
console.log(result.usage.billable_input_tokens);
```

Explicit `apiKey`, `baseURL`, and `defaultModel` override `ZILS_API_KEY`,
`ZILS_BASE_URL`, and `ZILS_DEFAULT_MODEL`. Defaults are
`https://training.zils.ai/decision` and `zils-shared`. Timeout is 60,000ms and
automatic retries are off; a repeated decision can incur another charge.
Browser use is disabled to avoid exposing API keys. CLI login files are not read.

`client.models.list()` returns model cards. `noul`, `choice`, and `score` preserve
question-specific answer types. `usage.input_tokens` counts model resources;
optional `usage.billable_input_tokens` is Zils's logical billing count.

[Complete guide](https://github.com/ooo-hq/zils/blob/main/docs/developer-tools.md).
