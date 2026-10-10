'use strict';

const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { createServer } = require('node:http');
const { once } = require('node:events');
const { mkdtempSync, readFileSync, rmSync } = require('node:fs');
const { EOL, tmpdir } = require('node:os');
const { join, resolve } = require('node:path');
const { test } = require('node:test');

const bin = resolve(process.env.ZILS_TEST_BIN || join(__dirname, '../bin/zils.cjs'));

function run(args, env, input = '') {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [bin, ...args], { env, stdio: ['pipe', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    child.stdout.on('data', data => { stdout += data; });
    child.stderr.on('data', data => { stderr += data; });
    child.on('error', reject);
    child.on('close', (code, signal) => resolve({ code, signal, stdout, stderr }));
    child.stdin.end(input);
  });
}

test('installed CLI bootstraps once, preserves stdin/JSON/exit codes, and reuses the cache offline', { timeout: 180000 }, async t => {
  const directory = mkdtempSync(join(tmpdir(), 'zils npm test '));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !/^(ZILS_|TYPESAFE_|PYTHON|PIP_)/.test(key)));
  if (process.env.ZILS_PYTHON) env.ZILS_PYTHON = process.env.ZILS_PYTHON;
  Object.assign(env, { XDG_CACHE_HOME: join(directory, 'cache'), ZILS_CONFIG_PATH: join(directory, 'config.json') });
  const boot = await Promise.all([run(['--version'], env), run(['train', 'submit', '--help'], env)]);
  for (const result of boot) assert.equal(result.code, 0, result.stderr);
  assert.equal(boot[0].stdout, `zils 0.1.0${EOL}`);
  assert.match(boot[1].stdout, /--min-brier-improvement/);
  assert.equal(boot.filter(result => result.stderr.includes('Setting up Zils')).length, 1);

  // An existing installation must not contact a package index again.
  env.HTTPS_PROXY = env.HTTP_PROXY = 'http://127.0.0.1:1';
  env.NO_PROXY = '127.0.0.1,localhost';
  const warm = await run(['--version'], env);
  assert.equal(warm.code, 0, warm.stderr);
  assert.equal(warm.stderr, '');

  const calls = [];
  const server = createServer(async (req, res) => {
    let text = ''; for await (const chunk of req) text += chunk;
    calls.push({ key: req.headers.authorization, path: req.url, body: text && JSON.parse(text) });
    res.setHeader('Content-Type', 'application/json');
    if (req.method === 'GET') res.end(JSON.stringify({ models: [{ name: 'zils-test', description: 'Test model', release_date: '2026-10-10' }] }));
    else res.end(JSON.stringify({ model: 'zils-test', answers: { refund: { type: 'noul', noul: 0.8 } }, usage: { input_tokens: 10, output_tokens: 0, billable_input_tokens: 7 } }));
  }).listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => { server.closeAllConnections(); server.close(); });
  const base = `http://127.0.0.1:${server.address().port}/decision`;
  const login = await run(['login', '--api-key', '-', '--base-url', base], env, 'test-only-key\n');
  assert.equal(login.code, 0, login.stderr);
  assert.ok(!JSON.stringify(login).includes('test-only-key'));
  assert.equal(JSON.parse(readFileSync(env.ZILS_CONFIG_PATH, 'utf8')).base_url, base);
  const models = await run(['models', '--json'], env);
  assert.equal(models.code, 0, models.stderr);
  assert.equal(JSON.parse(models.stdout).models[0].name, 'zils-test');
  const state = 'Refund $(do-not-execute) `also-not-code`; spaces "quotes"';
  const questions = JSON.stringify({ refund: { type: 'noul', instructions: 'Refund?' } });
  const decision = await run(['decide', '--state', '-', '--questions', questions, '--json'], env, state);
  assert.equal(decision.code, 0, decision.stderr);
  assert.equal(JSON.parse(decision.stdout).usage.billable_input_tokens, 7);
  assert.equal(calls.at(-1).body.state, state);
  assert.equal(calls.at(-1).key, 'Bearer test-only-key');
  const invalid = await run(['not-a-command'], env);
  assert.equal(invalid.code, 2);
  assert.equal(invalid.stdout, '');
  assert.equal((await run(['logout'], env)).code, 0);
  assert.equal((await run(['models', '--json'], env)).code, 2);
});

test('missing Python fails clearly before attempting setup', async t => {
  const directory = mkdtempSync(join(tmpdir(), 'zils no python '));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const result = await run(['--version'], { ...process.env, ZILS_PYTHON: join(directory, 'missing-python') });
  assert.equal(result.code, 2);
  assert.equal(result.stdout, '');
  assert.match(result.stderr, /Python 3.11\+/);
});
