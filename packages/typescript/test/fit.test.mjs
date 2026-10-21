import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { record } from '../dist/index.js';

test('record logs sync and async answers privately and leaves behaviour unchanged', async () => {
  const log = join(mkdtempSync(join(tmpdir(), 'zils-fit-')), 'fit-log.jsonl');
  process.env.ZILS_FIT_LOG = log;
  try {
    const urgent = record('urgent', ticket => ticket.includes('outage'));
    const team = record('team', async () => 'technical');
    const broken = record('broken', async () => { throw new Error('upstream down'); });
    assert.equal(urgent('outage now'), true);
    assert.equal(await team('outage now'), 'technical');
    await assert.rejects(broken('x'), /upstream down/);
    const lines = readFileSync(log, 'utf8').trim().split('\n').map(line => JSON.parse(line));
    assert.deepEqual(lines.map(({ question, state, answer }) => [question, state, answer]),
      [['urgent', 'outage now', true], ['team', 'outage now', 'technical']]);
    assert.equal(statSync(log).mode & 0o777, 0o600);
    const circular = {}; circular.self = circular;
    assert.equal(record('q', () => 7)(circular), 7);  // unloggable input never breaks the call
  } finally {
    delete process.env.ZILS_FIT_LOG;
  }
});
