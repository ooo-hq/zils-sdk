import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import { ZilsClient, noul, choice, score, APIError } from '../dist/index.js';

const response = {
  model: 'zils-test-r1',
  answers: {
    flag: { type: 'noul', noul: 0.75 },
    team: { type: 'choice', choice: 'billing', confidence: 0.8, probabilities: { billing: 0.8, other: 0.2 } },
    level: { type: 'score', score: 0.5, confidence: 0.5, probabilities: { 0: 0.5, 1: 0.5 }, legend: { 0: { name: 'low' }, 1: ['high'] } },
  },
  usage: { input_tokens: 100, output_tokens: 0, billable_input_tokens: 73 },
};
const questions = {
  flag: noul('Is a refund requested?'),
  team: choice('Which team?', { billing: 'Billing', other: null }),
  level: score('Urgency?', [{ name: 'low' }, ['high']]),
};

async function serve(t, handler) {
  const server = createServer(handler).listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return `http://127.0.0.1:${server.address().port}`;
}

test('calls the configured Zils endpoint and retains typed answers and billable usage', async t => {
  const received = [];
  const baseURL = await serve(t, async (req, res) => {
    let text = ''; for await (const chunk of req) text += chunk;
    received.push({ path: req.url, key: req.headers.authorization, body: text && JSON.parse(text) });
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify(req.method === 'GET' ? { models: [{ name: 'zils-shared', description: 'Shared', release_date: '2026-10-04' }] } : response));
  });
  const client = new ZilsClient({ apiKey: 'test-key', baseURL: `${baseURL}/decision/` });
  assert.equal((await client.models.list())[0].name, 'zils-shared');
  const result = await client.systemOne({ state: { ticket: 'Refund' }, questions });
  assert.equal(result.usage.billable_input_tokens, 73);
  assert.equal(result.answers.flag.noul, 0.75);
  assert.deepEqual(result.answers.level.legend, { 0: { name: 'low' }, 1: ['high'] });
  assert.equal(received[1].body.model, 'zils-shared');
  assert.equal(received[1].key, 'Bearer test-key');
  assert.equal(received[0].path, '/decision/v1/models');
  assert.equal(received[1].path, '/decision/v1/systemone');
});

test('Zils configuration wins over TypeSafe environment settings', t => {
  const values = {
    ZILS_API_KEY: 'zils-key', ZILS_BASE_URL: 'https://example.com/decision', ZILS_DEFAULT_MODEL: 'private-model',
    TYPESAFE_API_KEY: 'wrong-key', TYPESAFE_BASE_URL: 'https://wrong.example', TYPESAFE_DEFAULT_MODEL: 'wrong-model',
  };
  for (const [key, value] of Object.entries(values)) {
    const previous = process.env[key]; process.env[key] = value;
    t.after(() => previous === undefined ? delete process.env[key] : process.env[key] = previous);
  }
  assert.equal(new ZilsClient().baseURL, 'https://example.com/decision');
  assert.equal(new ZilsClient().defaultModel, 'private-model');
  assert.equal(new ZilsClient({ defaultModel: 'explicit' }).defaultModel, 'explicit');
});

test('missing keys and unsafe explicit settings are rejected without echoing credentials', t => {
  const previous = process.env.ZILS_API_KEY; delete process.env.ZILS_API_KEY;
  t.after(() => previous === undefined ? delete process.env.ZILS_API_KEY : process.env.ZILS_API_KEY = previous);
  assert.throws(() => new ZilsClient(), /ZILS_API_KEY/);
  for (const options of [
    { apiKey: '' }, { apiKey: 'private-secret\nheader' }, { defaultModel: '' },
    { baseURL: 'http://remote.example' }, { baseURL: 'http://127.example.com' },
    { baseURL: 'https://user:private-secret@example.com' },
    { baseURL: 'https://example.com?secret=private-secret' }, { baseURL: 'https://example.com/#fragment' },
  ]) {
    assert.throws(() => new ZilsClient({ apiKey: 'test-key', ...options }), error => !error.message.includes('private-secret'));
  }
});

test('a rate limit reaches the caller without an automatic second paid attempt', async t => {
  let attempts = 0;
  const baseURL = await serve(t, (req, res) => {
    attempts++; res.writeHead(429, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { code: 'rate_limit', message: 'Limit reached' } }));
  });
  await assert.rejects(new ZilsClient({ apiKey: 'key', baseURL }).systemOne({ state: 'x', questions }), error => error instanceof APIError && error.status === 429);
  assert.equal(attempts, 1);
});

test('redirects cannot forward a decision or credentials to another endpoint', async t => {
  let forwarded = 0;
  const target = await serve(t, (req, res) => { forwarded++; res.end(JSON.stringify(response)); });
  const baseURL = await serve(t, (req, res) => { res.writeHead(307, { Location: `${target}/stolen` }); res.end(); });
  await assert.rejects(new ZilsClient({ apiKey: 'key', baseURL }).systemOne({ state: 'private', questions }));
  assert.equal(forwarded, 0);
});

test('an undefined optional retry count still makes only one paid attempt', async t => {
  let attempts = 0;
  const baseURL = await serve(t, (req, res) => {
    attempts++; res.writeHead(429, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { code: 'rate_limit', message: 'Limit reached' } }));
  });
  const client = new ZilsClient({ apiKey: 'key', baseURL, retry: {
    maxRetries: undefined, backoffInitialMs: 1, backoffMaxMs: 1,
  } });
  await assert.rejects(client.systemOne({ state: 'x', questions }));
  assert.equal(attempts, 1);
});

test('CommonJS consumers get the same client and question helpers', () => {
  const require = createRequire(import.meta.url);
  assert.equal(typeof require('../dist/index.js').ZilsClient, 'function');
  assert.equal(require('../dist/index.js').noul('A?').type, 'noul');
});
