#!/usr/bin/env node
'use strict';

const { spawn, spawnSync } = require('node:child_process');
const { join } = require('node:path');

// Use an argument array throughout: dataset paths and questions are never shell code.
const candidates = process.env.ZILS_PYTHON
  ? [[process.env.ZILS_PYTHON]]
  : process.platform === 'win32' ? [['py', '-3'], ['python'], ['python3']] : [['python3'], ['python']];
const python = candidates.find(([command, ...prefix]) => {
  const result = spawnSync(command, [...prefix, '-I', '-c',
    'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'],
  { stdio: 'ignore', timeout: 10000, windowsHide: true });
  return result.status === 0;
});

if (!python) {
  console.error('Zils requires Python 3.11+. Install it from https://www.python.org/downloads/ and try again.');
  console.error('If Python is elsewhere, set ZILS_PYTHON to its executable path.');
  process.exitCode = 2;
} else {
  const [command, ...prefix] = python;
  const child = spawn(command, [...prefix, '-I', join(__dirname, '..', 'bootstrap.py'), ...process.argv.slice(2)],
    { stdio: 'inherit' });
  for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal));
  child.on('error', () => { console.error('Could not start the Zils Python runtime.'); process.exitCode = 2; });
  child.on('exit', (code, signal) => { process.exitCode = code ?? (signal === 'SIGINT' ? 130 : 143); });
}
