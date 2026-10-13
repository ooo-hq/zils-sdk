import assert from 'node:assert/strict';
import { once } from 'node:events';
import { readFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { ZilsClient, APIError } from '../dist/index.js';

const cases = JSON.parse(readFileSync(new URL('../../../tests/fixtures/ranking.json', import.meta.url)));
const groupedCases = JSON.parse(readFileSync(new URL('../../../tests/fixtures/grouped-ranking.json', import.meta.url)));

async function fixture(t, scenario = cases[0], transform = () => {}, status = 200) {
  const calls = [];
  const server = createServer(async (req, res) => {
    let text = ''; for await (const chunk of req) text += chunk;
    const body = JSON.parse(text);
    calls.push({ path: req.url, authorization: req.headers.authorization, body });
    const answers = {};
    for (const [name, question] of Object.entries(body.questions)) {
      if (scenario.strengths || scenario.position_probabilities) {
        const weights = scenario.position_probabilities || Object.values(question.criteria).map(option =>
          scenario.strengths[scenario.items.findIndex(item => JSON.stringify(item) === JSON.stringify(option.item))]);
        const total = weights.reduce((sum, value) => sum + value, 0);
        const probabilities = Object.fromEntries(Object.keys(question.criteria).map((key, i) => [key, weights[i] / total]));
        const choice = Object.keys(probabilities).reduce((a, b) => probabilities[a] >= probabilities[b] ? a : b);
        answers[name] = { type: 'choice', choice, probabilities, confidence: probabilities[choice] };
        continue;
      }
      const a = scenario.items.findIndex(item => JSON.stringify(item) === JSON.stringify(question.criteria.a.item));
      const b = scenario.items.findIndex(item => JSON.stringify(item) === JSON.stringify(question.criteria.b.item));
      const p = scenario.probabilities[a][b];
      answers[name] = { type: 'choice', choice: p >= 0.5 ? 'a' : 'b',
        probabilities: { a: p, b: 1 - p }, confidence: Math.abs(2 * p - 1) };
    }
    const value = { model: 'zils-test-r1', answers,
      usage: { input_tokens: 1234, output_tokens: 0, billable_input_tokens: 456 } };
    transform(value);
    res.writeHead(status, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(status === 200 ? value : { error: { code: 'rate_limit', message: 'Limit reached' } }));
  }).listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const client = new ZilsClient({ apiKey: 'test-key', baseURL: `http://127.0.0.1:${server.address().port}` });
  assert.equal(typeof client.rank, 'function', 'SDK rank method is missing');
  return { client, calls };
}

function assertRanking(result, expected) {
  assert.deepEqual(result.ranking.map(item => item.index), expected.map(row => row[0]));
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(result.ranking[i].score - expected[i][1]) < 1e-6);
  }
}

for (const scenario of cases) {
  test(scenario.name, async t => {
    const { client, calls } = await fixture(t, scenario);
    const result = await client.rank({ items: scenario.items, instructions: 'Most useful first', model: 'private-model' });
    assertRanking(result, scenario.ranking);
    assert.equal(result.model, 'zils-test-r1');
    assert.equal(result.comparisons, 6);
    assert.deepEqual(result.usage, { input_tokens: 1234, output_tokens: 0, billable_input_tokens: 456 });
    assert.equal(calls.length, 1);
    assert.equal(calls[0].path, '/v1/systemone');
    assert.equal(calls[0].authorization, 'Bearer test-key');
    assert.equal(calls[0].body.model, 'private-model');
    assert.deepEqual(calls[0].body.state, { criterion: 'Most useful first' });
    const pairs = Object.values(calls[0].body.questions).map(q => [q.criteria.a.item, q.criteria.b.item]);
    assert.equal(new Set(pairs.map(pair => JSON.stringify(pair))).size, 6);
    assert.ok(pairs.every(([a, b]) => a !== b && pairs.some(([c, d]) => a === d && b === c)));
  });
}

test('reordered inputs retain the ranking with updated original indices', async t => {
  const { client } = await fixture(t);
  const result = await client.rank({ items: ['high', 'low', 'medium'], instructions: 'Most useful first' });
  assertRanking(result, [[0, 1], [2, 0.5], [1, 0.25]]);
});

for (const scenario of groupedCases) {
  test(`grouped: ${scenario.name}`, async t => {
    const { client, calls } = await fixture(t, scenario);
    const result = await client.rank({ items: scenario.items, instructions: 'Most useful first',
      model: 'private-model', groupSize: scenario.group_size });
    assertRanking(result, scenario.ranking);
    assert.equal(result.comparisons, scenario.items.length);
    assert.deepEqual(result.usage, { input_tokens: 1234, output_tokens: 0, billable_input_tokens: 456 });
    assert.equal(calls.length, 1);
    const body = calls[0].body;
    assert.equal(body.model, 'private-model');
    assert.deepEqual(body.state, { criterion: 'Most useful first' });
    const groups = Object.values(body.questions).map(q => Object.values(q.criteria));
    const size = Math.min(scenario.group_size, scenario.items.length);
    assert.ok(groups.every(group => group.length === size));
    for (let position = 0; position < size; position++) {
      assert.deepEqual(groups.map(group => JSON.stringify(group[position].item)).sort(),
        scenario.items.map(item => JSON.stringify(item)).sort());
    }
  });
}

test('grouped requests snapshot items and group size before waiting', async t => {
  const request = { items: ['high', 'low', 'medium'], instructions: 'Rank', groupSize: 4 };
  const { client } = await fixture(t, groupedCases[2], () => {
    request.items.length = 0; request.groupSize = 2;
  });
  const result = await client.rank(request);
  assertRanking(result, [[0, 1], [2, 0.5], [1, 0.25]]);
  assert.equal(result.comparisons, 3);
});

test('invalid group sizes never send a paid request', async t => {
  const { client, calls } = await fixture(t);
  for (const groupSize of [0, 1, 11, 2.5, true, '4', null, NaN, Infinity]) {
    await assert.rejects(client.rank({ items: ['a', 'b', 'c'], instructions: 'Rank', groupSize }));
  }
  assert.equal(calls.length, 0);
});

test('grouped incomplete or invalid answers fail without retry', async t => {
  for (const transform of [
    v => { delete v.answers[Object.keys(v.answers)[0]]; },
    v => { v.answers.extra = Object.values(v.answers)[0]; },
    v => {
      const p = Object.values(v.answers)[0].probabilities;
      const key = Object.keys(p)[0]; p.unexpected = p[key]; delete p[key];
    },
    v => { Object.values(v.answers)[0].choice = 'unexpected'; },
    v => { Object.values(v.answers)[0].confidence = -1; },
    v => { Object.values(v.answers)[0].type = 'noul'; },
    v => { Object.values(v.answers)[0].probabilities.a = 2; },
  ]) {
    const { client, calls } = await fixture(t, groupedCases[0], transform);
    await assert.rejects(client.rank({ items: groupedCases[0].items, instructions: 'Rank', groupSize: 4 }));
    assert.equal(calls.length, 1);
  }
});

test('grouped zero and one probabilities keep finite strengths', async t => {
  const { client } = await fixture(t, groupedCases[0], value => {
    for (const answer of Object.values(value.answers)) {
      answer.probabilities = Object.fromEntries(Object.keys(answer.probabilities).map(key => [key, Number(key === answer.choice)]));
    }
  });
  const result = await client.rank({ items: groupedCases[0].items, instructions: 'Rank', groupSize: 4 });
  assert.equal(result.ranking[0].index, 9);
  assert.ok(result.ranking.every(r => Number.isFinite(r.score) && r.score > 0 && r.score <= 1));
});

test('invalid inputs are rejected before sending a paid request', async t => {
  const { client, calls } = await fixture(t);
  const circular = []; circular.push(circular);
  for (const [items, instructions] of [
    [[], 'Rank'], [['a'], 'Rank'], [Array(11).fill('a'), 'Rank'], ['ab', 'Rank'],
    [['a', 'b'], ' '], [['a', 'b'], null], [['a', NaN], 'Rank'],
    [['a', undefined], 'Rank'], [['a', circular], 'Rank'], [['a', 1n], 'Rank'],
  ]) {
    await assert.rejects(client.rank({ items, instructions }));
  }
  assert.equal(calls.length, 0);
});

test('non-JSON object instances are rejected without lossy conversion', async () => {
  let calls = 0;
  const client = new ZilsClient({ apiKey: 'test-key', fetch: async () => {
    calls++;
    return new Response('{}', { headers: { 'Content-Type': 'application/json' } });
  } });
  for (const item of [new Map([['key', 'value']]), new Set(['value']), new Date(), { nested: new Map() }]) {
    await assert.rejects(client.rank({ items: [item, 'b'], instructions: 'Most useful first' }));
  }
  assert.equal(calls, 0);
});

test('ten items use ninety comparisons in one request', async t => {
  const scenario = { items: Array.from({ length: 10 }, (_, i) => i),
    probabilities: Array.from({ length: 10 }, () => Array(10).fill(0.5)) };
  const { client, calls } = await fixture(t, scenario);
  const result = await client.rank({ items: scenario.items, instructions: 'Most useful first' });
  assert.equal(result.comparisons, 90);
  assert.equal(Object.keys(calls[0].body.questions).length, 90);
  assertRanking(result, scenario.items.map(i => [i, 1]));
});

test('zero and one probabilities produce finite strengths', async t => {
  const { client } = await fixture(t, { items: ['bad', 'good'], probabilities: [[null, 0], [1, null]] });
  const result = await client.rank({ items: ['bad', 'good'], instructions: 'Most useful first' });
  assert.deepEqual(result.ranking.map(item => item.index), [1, 0]);
  assert.equal(result.ranking[0].score, 1);
  assert.ok(result.ranking.every(item => Number.isFinite(item.score) && item.score > 0 && item.score <= 1));
});

test('ten decisive items converge without collapsing small scores', async t => {
  const items = Array.from({ length: 10 }, (_, i) => i);
  const { client } = await fixture(t, { items,
    probabilities: items.map(i => items.map(j => Number(i > j))) });
  const result = await client.rank({ items, instructions: 'Most useful first' });
  assert.deepEqual(result.ranking.map(item => item.index), [...items].reverse());
  assert.ok(result.ranking.every(item => Number.isFinite(item.score) && item.score > 0 && item.score <= 1));
});

test('input list is snapshotted before waiting for the response', async t => {
  const items = [...cases[0].items];
  const { client } = await fixture(t, cases[0], () => { items.pop(); });
  const result = await client.rank({ items, instructions: 'Most useful first' });
  assertRanking(result, cases[0].ranking);
});

test('structured duplicate items keep their original indices', async t => {
  const items = [{ text: '返金', tags: [true, null] }, { text: '返金', tags: [true, null] }];
  const { client, calls } = await fixture(t, { items, probabilities: [[0.5, 0.5], [0.5, 0.5]] });
  const result = await client.rank({ items, instructions: 'Most useful first' });
  assertRanking(result, [[0, 1], [1, 1]]);
  assert.deepEqual(calls[0].body.questions.pair_0_1.criteria.a.item, items[0]);
});

test('a pre-cancelled request makes no paid call', async t => {
  const { client, calls } = await fixture(t);
  await assert.rejects(client.rank({ items: cases[0].items, instructions: 'Most useful first' },
    { signal: AbortSignal.abort() }));
  assert.equal(calls.length, 0);
});

test('incomplete or invalid probabilities never produce a ranking', async t => {
  for (const transform of [
    value => { delete value.answers[Object.keys(value.answers)[0]]; },
    value => { delete Object.values(value.answers)[0].confidence; },
    value => { Object.values(value.answers)[0].confidence = -1; },
    ...[{ a: 0.9 }, { a: -0.1, b: 1.1 }, { a: 0.8, b: 0.8 }].map(probabilities =>
      value => { Object.values(value.answers)[0].probabilities = probabilities; }),
  ]) {
    const { client, calls } = await fixture(t, cases[0], transform);
    await assert.rejects(client.rank({ items: cases[0].items, instructions: 'Most useful first' }));
    assert.equal(calls.length, 1);
  }
});

test('missing billable usage is not substituted with resource usage', async t => {
  const { client } = await fixture(t, cases[0], value => { delete value.usage.billable_input_tokens; });
  const result = await client.rank({ items: cases[0].items, instructions: 'Most useful first' });
  assert.equal(result.usage.billable_input_tokens, undefined);
  assert.equal(result.usage.input_tokens, 1234);
});

test('missing or invalid model and usage metadata cannot produce a result', async t => {
  for (const transform of [
    value => { delete value.model; }, value => { value.model = ''; },
    value => { delete value.usage; }, value => { delete value.usage.input_tokens; },
    value => { value.usage.input_tokens = -1; }, value => { value.usage.output_tokens = 0.5; },
    value => { value.usage.billable_input_tokens = 'bad'; },
  ]) {
    const { client, calls } = await fixture(t, cases[0], transform);
    await assert.rejects(client.rank({ items: cases[0].items, instructions: 'Most useful first' }));
    assert.equal(calls.length, 1);
  }
});

test('rate limits propagate without another paid attempt', async t => {
  const { client, calls } = await fixture(t, cases[0], () => {}, 429);
  await assert.rejects(client.rank({ items: cases[0].items, instructions: 'Most useful first' }),
    error => error instanceof APIError && error.status === 429);
  assert.equal(calls.length, 1);
});
