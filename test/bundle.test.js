'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const test = require('node:test');
const { BundleValidationError, validateBundle } = require('../lib/bundle');
const { makeBundle } = require('./helpers');

test('the published bundle declares only brainx-install', async () => {
  const bundle = await validateBundle(path.resolve(__dirname, '..'));
  assert.equal(bundle.version, '1.0.1');
  assert.deepEqual(bundle.skills.map((skill) => skill.name), ['brainx-install']);
  const packageJson = JSON.parse(
    await fs.readFile(path.resolve(__dirname, '..', 'package.json'), 'utf8'),
  );
  assert.equal(packageJson.files.includes('skills/brainx-install/'), true);
  assert.equal(packageJson.files.includes('skills/'), false);
});

test('manifest validation rejects duplicate and unsafe names', async (t) => {
  const packageRoot = await makeBundle(t);
  await fs.writeFile(
    path.join(packageRoot, 'manifest.json'),
    JSON.stringify({ schemaVersion: 1, skills: ['alpha', 'alpha', '../escape'] }),
  );
  await assert.rejects(validateBundle(packageRoot), BundleValidationError);
});

test('manifest validation rejects missing declared skill directories', async (t) => {
  const packageRoot = await makeBundle(t);
  await fs.rm(path.join(packageRoot, 'skills', 'beta'), { recursive: true });
  await assert.rejects(
    validateBundle(packageRoot),
    /Manifest declares missing skill directories: beta/,
  );
});

test('bundle validation rejects symbolic links', async (t) => {
  const packageRoot = await makeBundle(t);
  await fs.symlink(
    path.join(packageRoot, 'skills', 'alpha', 'SKILL.md'),
    path.join(packageRoot, 'skills', 'alpha', 'linked.md'),
  );
  await assert.rejects(validateBundle(packageRoot), /regular files and directories/);
});
