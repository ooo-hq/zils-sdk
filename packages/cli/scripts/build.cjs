'use strict';

const { spawnSync } = require('node:child_process');
const { copyFileSync, mkdirSync, readFileSync, rmSync } = require('node:fs');
const { join } = require('node:path');
const root = join(__dirname, '..');
const source = join(root, '..', 'python');
const version = require('../package.json').version;
const pyproject = readFileSync(join(source, 'pyproject.toml'), 'utf8');
if (!pyproject.includes(`version = "${version}"`)) throw new Error('CLI and Python package versions must match.');
const vendor = join(root, 'vendor');
rmSync(vendor, { recursive: true, force: true });
mkdirSync(vendor, { recursive: true });
const result = spawnSync(process.env.ZILS_PYTHON || 'python3',
  ['-m', 'build', '--wheel', '--outdir', vendor, source], { stdio: 'inherit' });
if (result.status !== 0) process.exit(result.status || 1);
for (const name of ['LICENSE', 'THIRD_PARTY_NOTICES.md']) copyFileSync(join(source, name), join(root, name));
