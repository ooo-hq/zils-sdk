import assert from 'node:assert/strict';
import { ZilsClient, noul, choice, score, APIError } from '../dist/index.js';

const client = new ZilsClient({ apiKey: 'valid-key', baseURL: process.argv[2] });
assert.deepEqual(new Set((await client.models.list()).map(model => model.name)), new Set(['zils-shared', 'zils-test-r1']));
const result = await client.systemOne({
  state: { ticket: 'Refund please' },
  questions: {
    flag: noul('Refund?'),
    team: choice('Team?', { billing: null, support: null }),
    level: score(null, [{ level: 'low' }, ['high']]),
  },
});
assert.equal(result.model, 'zils-test-r1');
assert.equal(result.answers.flag.noul, 0.5);
assert.deepEqual(result.answers.level.legend, { 0: { level: 'low' }, 1: ['high'] });
assert.equal(result.usage.billable_input_tokens, 73);
const ranked = await client.rank({ items: ['a', 'b', 'c'], instructions: 'Most useful first' });
assert.deepEqual(ranked.ranking, [{ index: 0, score: 1 }, { index: 1, score: 1 }, { index: 2, score: 1 }]);
assert.equal(ranked.model, 'zils-test-r1');
assert.equal(ranked.comparisons, 6);
assert.equal(ranked.usage.billable_input_tokens, 73);
await assert.rejects(new ZilsClient({ apiKey: 'invalid', baseURL: process.argv[2] }).models.list(), error => error instanceof APIError && error.status === 401);
console.log('TypeScript SDK passed the Zils HTTP contract checks.');
