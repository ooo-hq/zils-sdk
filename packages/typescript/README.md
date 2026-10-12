# Zils TypeScript SDK

Node.js 20+. Typed decisions using the existing Zils API and TypeSafe-compatible
questions. Install it in your application:

```sh
npm install @zils/sdk
```

Set the server's `ZILS_API_KEY`, then use either ESM `import` or CommonJS `require`:

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

## Rank a short list

Version 0.2.0 adds experimental `client.rank`. Install with `npm install @zils/sdk`:

```ts
import { ZilsClient } from '@zils/sdk';

const client = new ZilsClient();
const items = ['Update the logo', 'Checkout is failing', 'Fix a typo'];
const result = await client.rank({ items, instructions: 'Most urgent customer impact first' });
for (const entry of result.ranking) {
  console.log(items[entry.index], entry.score);
}
console.log(result.comparisons, result.usage.billable_input_tokens);
```

Accepts 2–10 JSON items. One existing decision API request compares every pair in
both orders: `n * (n - 1)` model judgments, up to 90. It fits Bradley–Terry strengths
locally and returns original indices, relative scores, the actual model, and
unmodified usage. A score of 1 means strongest in this set, not certainty or
absolute quality. Equal scores retain input order. This is an SDK helper, not a
new model or HTTP endpoint. Existing context, timeout, billing, and retry rules
apply; one HTTP request does not mean one model evaluation. See the
[ranking contract](https://github.com/ooo-hq/zils-sdk/blob/main/docs/developer-tools.md#ranking)
for numerical limits, failure behavior, and examples.

Evaluate ranking on your own task before relying on the order. Use ordinary code
for arithmetic, unit conversion, and purely numeric sorting; the model can make
incorrect comparisons, and fitting strengths does not correct that evidence.

[Complete guide](https://github.com/ooo-hq/zils-sdk/blob/main/docs/developer-tools.md).
