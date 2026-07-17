'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const test = require('node:test');
const { runInstaller } = require('../lib/installer');
const {
  captureStream,
  makeBundle,
  makeHome,
  readJson,
  sequentialUuid,
} = require('./helpers');

async function run(command, packageRoot, homeDir, overrides = {}) {
  const stdout = captureStream();
  const stderr = captureStream();
  const status = await runInstaller(command, {
    packageRoot,
    homeDir,
    stdout: stdout.stream,
    stderr: stderr.stream,
    now: () => overrides.timestamp || '2026-07-17T00:00:00.000Z',
    uuid: overrides.uuid || sequentialUuid(command),
    ...overrides,
  });
  return { status, stdout: stdout.text(), stderr: stderr.text() };
}

function receiptPath(homeDir) {
  return path.join(homeDir, '.brainx', 'receipt.json');
}

function skillPath(homeDir, adapter, skill, file = 'SKILL.md') {
  const root = adapter === 'codex' ? ['.agents', 'skills'] : ['.claude', 'skills'];
  return path.join(homeDir, ...root, skill, file);
}

test('fresh install copies every skill to both destinations and writes ownership', async (t) => {
  const packageRoot = await makeBundle(t);
  const homeDir = await makeHome(t);
  const result = await run('install', packageRoot, homeDir);

  assert.equal(result.status, 0);
  assert.match(result.stdout, /BrainX skills 1\.0\.0 installed for Codex/);
  assert.match(result.stdout, /BrainX skills 1\.0\.0 installed for Claude Code/);
  assert.equal(result.stderr, '');
  assert.match(await fs.readFile(skillPath(homeDir, 'codex', 'alpha'), 'utf8'), /1\.0\.0/);
  assert.match(await fs.readFile(skillPath(homeDir, 'claude', 'beta'), 'utf8'), /1\.0\.0/);

  const receipt = await readJson(receiptPath(homeDir));
  assert.equal(receipt.packageName, 'brainx-skills');
  assert.equal(receipt.adapters.codex.packageVersion, '1.0.0');
  assert.equal(receipt.adapters.claude.packageVersion, '1.0.0');
  assert.deepEqual(receipt.adapters.codex.skills.map((skill) => skill.name), ['alpha', 'beta']);
});

test('repeated install is idempotent and does not rewrite the receipt', async (t) => {
  const packageRoot = await makeBundle(t);
  const homeDir = await makeHome(t);
  assert.equal((await run('install', packageRoot, homeDir)).status, 0);
  const before = await fs.readFile(receiptPath(homeDir), 'utf8');

  const result = await run('install', packageRoot, homeDir, {
    timestamp: '2026-07-18T00:00:00.000Z',
  });
  const after = await fs.readFile(receiptPath(homeDir), 'utf8');

  assert.equal(result.status, 0);
  assert.equal(result.stdout, '\u2713 BrainX skills 1.0.0 are already installed\n');
  assert.equal(result.stderr, '');
  assert.equal(after, before);
});

test('update replaces a complete older managed release', async (t) => {
  const oldBundle = await makeBundle(t, { version: '0.9.0', contentTag: 'old release' });
  const newBundle = await makeBundle(t, { version: '1.0.0', contentTag: 'new release' });
  const homeDir = await makeHome(t);
  assert.equal((await run('install', oldBundle, homeDir)).status, 0);

  const result = await run('update', newBundle, homeDir);
  assert.equal(result.status, 0);
  assert.match(await fs.readFile(skillPath(homeDir, 'codex', 'alpha'), 'utf8'), /new release/);
  assert.match(await fs.readFile(skillPath(homeDir, 'claude', 'alpha'), 'utf8'), /new release/);
  const receipt = await readJson(receiptPath(homeDir));
  assert.equal(receipt.lastOperation.command, 'update');
  assert.equal(receipt.adapters.codex.packageVersion, '1.0.0');
  assert.equal(receipt.adapters.claude.packageVersion, '1.0.0');
});

test('update with no managed installation changes nothing', async (t) => {
  const homeDir = await makeHome(t);
  const result = await run('update', path.join(homeDir, 'missing package'), homeDir);

  assert.equal(result.status, 0);
  assert.equal(
    result.stdout,
    'No BrainX-managed skills are currently installed.\nRun: npx brainx-skills install\n',
  );
  assert.equal(result.stderr, '');
  await assert.rejects(fs.lstat(path.join(homeDir, '.agents')), { code: 'ENOENT' });
  await assert.rejects(fs.lstat(path.join(homeDir, '.claude')), { code: 'ENOENT' });
});

test('an unmanaged Codex conflict is preserved while Claude installs successfully', async (t) => {
  const packageRoot = await makeBundle(t);
  const homeDir = await makeHome(t);
  const conflict = skillPath(homeDir, 'codex', 'alpha');
  await fs.mkdir(path.dirname(conflict), { recursive: true });
  await fs.writeFile(conflict, 'user-authored\n');

  const result = await run('install', packageRoot, homeDir);
  assert.equal(result.status, 1);
  assert.equal(await fs.readFile(conflict, 'utf8'), 'user-authored\n');
  assert.match(await fs.readFile(skillPath(homeDir, 'claude', 'alpha'), 'utf8'), /1\.0\.0/);
  assert.match(result.stderr, /Codex installation failed/);
  assert.match(result.stderr, /cannot prove ownership/);
  assert.match(result.stderr, new RegExp(path.dirname(conflict).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));

  const receipt = await readJson(receiptPath(homeDir));
  assert.equal(receipt.adapters.codex, undefined);
  assert.equal(receipt.adapters.claude.packageVersion, '1.0.0');
});

test('a forged ownership path invalidates the receipt and protects all destinations', async (t) => {
  const packageRoot = await makeBundle(t);
  const homeDir = await makeHome(t);
  const outside = path.join(homeDir, 'user data', 'alpha');
  await fs.mkdir(outside, { recursive: true });
  await fs.writeFile(path.join(outside, 'keep.txt'), 'keep\n');
  await fs.mkdir(path.dirname(receiptPath(homeDir)), { recursive: true });
  await fs.writeFile(receiptPath(homeDir), JSON.stringify({
    schemaVersion: 1,
    packageName: 'brainx-skills',
    adapters: {
      codex: {
        label: 'Codex',
        destination: path.join(homeDir, '.agents', 'skills'),
        packageVersion: '0.9.0',
        skills: [{ name: 'alpha', directory: outside, digest: '0'.repeat(64) }],
      },
    },
  }));

  const result = await run('install', packageRoot, homeDir);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Unsafe Codex owned path/);
  assert.equal(await fs.readFile(path.join(outside, 'keep.txt'), 'utf8'), 'keep\n');
  await assert.rejects(fs.lstat(path.join(homeDir, '.claude')), { code: 'ENOENT' });
});

test('malformed receipts fail closed without changing skill directories', async (t) => {
  const packageRoot = await makeBundle(t);
  const homeDir = await makeHome(t);
  await fs.mkdir(path.dirname(receiptPath(homeDir)), { recursive: true });
  await fs.writeFile(receiptPath(homeDir), '{not json');

  const result = await run('install', packageRoot, homeDir);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Invalid JSON in BrainX receipt/);
  await assert.rejects(fs.lstat(path.join(homeDir, '.agents')), { code: 'ENOENT' });
  await assert.rejects(fs.lstat(path.join(homeDir, '.claude')), { code: 'ENOENT' });
});

test('symlinked destination components fail safely', async (t) => {
  const packageRoot = await makeBundle(t);
  const homeDir = await makeHome(t);
  const outside = path.join(path.dirname(homeDir), 'outside skills');
  await fs.mkdir(outside);
  await fs.mkdir(path.join(homeDir, '.agents'));
  await fs.symlink(outside, path.join(homeDir, '.agents', 'skills'));

  const result = await run('install', packageRoot, homeDir);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /Symbolic-link destinations are not supported/);
  assert.deepEqual(await fs.readdir(outside), []);
  assert.match(await fs.readFile(skillPath(homeDir, 'claude', 'alpha'), 'utf8'), /1\.0\.0/);
});

test('a checksum mismatch in an owned skill is repaired', async (t) => {
  const packageRoot = await makeBundle(t);
  const homeDir = await makeHome(t);
  assert.equal((await run('install', packageRoot, homeDir)).status, 0);
  await fs.writeFile(skillPath(homeDir, 'codex', 'alpha'), 'locally modified\n');

  const result = await run('install', packageRoot, homeDir);
  assert.equal(result.status, 0);
  assert.match(result.stdout, /installed for Codex/);
  assert.match(result.stdout, /already installed for Claude Code/);
  assert.match(await fs.readFile(skillPath(homeDir, 'codex', 'alpha'), 'utf8'), /1\.0\.0/);
});

test('staged validation failure preserves that adapter while the other adapter updates', async (t) => {
  const oldBundle = await makeBundle(t, { version: '0.9.0', contentTag: 'old release' });
  const newBundle = await makeBundle(t, { version: '1.0.0', contentTag: 'new release' });
  const homeDir = await makeHome(t);
  assert.equal((await run('install', oldBundle, homeDir)).status, 0);

  const result = await run('update', newBundle, homeDir, {
    copySkill: async (source, target, context) => {
      await fs.cp(source, target, { recursive: true });
      if (context.adapter.id === 'codex' && context.skill.name === 'alpha') {
        await fs.writeFile(path.join(target, 'SKILL.md'), 'corrupt stage\n');
      }
    },
  });

  assert.equal(result.status, 1);
  assert.match(await fs.readFile(skillPath(homeDir, 'codex', 'alpha'), 'utf8'), /old release/);
  assert.match(await fs.readFile(skillPath(homeDir, 'claude', 'alpha'), 'utf8'), /new release/);
  const receipt = await readJson(receiptPath(homeDir));
  assert.equal(receipt.adapters.codex.packageVersion, '0.9.0');
  assert.equal(receipt.adapters.claude.packageVersion, '1.0.0');
});

test('receipt write failure rolls back both destination replacements', async (t) => {
  const oldBundle = await makeBundle(t, { version: '0.9.0', contentTag: 'old release' });
  const newBundle = await makeBundle(t, { version: '1.0.0', contentTag: 'new release' });
  const homeDir = await makeHome(t);
  assert.equal((await run('install', oldBundle, homeDir)).status, 0);
  const oldReceipt = await fs.readFile(receiptPath(homeDir), 'utf8');

  const result = await run('update', newBundle, homeDir, {
    writeReceipt: async () => {
      throw new Error('simulated receipt failure');
    },
  });

  assert.equal(result.status, 1);
  assert.match(result.stderr, /Installed changes were rolled back/);
  assert.match(await fs.readFile(skillPath(homeDir, 'codex', 'alpha'), 'utf8'), /old release/);
  assert.match(await fs.readFile(skillPath(homeDir, 'claude', 'alpha'), 'utf8'), /old release/);
  assert.equal(await fs.readFile(receiptPath(homeDir), 'utf8'), oldReceipt);
});

test('update removes obsolete skills only when the receipt proves ownership', async (t) => {
  const oldBundle = await makeBundle(t, {
    version: '0.9.0',
    skills: ['alpha', 'obsolete'],
    contentTag: 'old release',
  });
  const newBundle = await makeBundle(t, {
    version: '1.0.0',
    skills: ['alpha'],
    contentTag: 'new release',
  });
  const homeDir = await makeHome(t);
  assert.equal((await run('install', oldBundle, homeDir)).status, 0);

  const result = await run('update', newBundle, homeDir);
  assert.equal(result.status, 0);
  await assert.rejects(fs.lstat(path.dirname(skillPath(homeDir, 'codex', 'obsolete'))), { code: 'ENOENT' });
  await assert.rejects(fs.lstat(path.dirname(skillPath(homeDir, 'claude', 'obsolete'))), { code: 'ENOENT' });
  const receipt = await readJson(receiptPath(homeDir));
  assert.deepEqual(receipt.adapters.codex.skills.map((skill) => skill.name), ['alpha']);
  assert.deepEqual(receipt.adapters.claude.skills.map((skill) => skill.name), ['alpha']);
});
