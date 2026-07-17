'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const test = require('node:test');
const claude = require('../adapters/claude');
const codex = require('../adapters/codex');
const cursor = require('../adapters/cursor');
const {
  NonInteractiveInstallError,
  detectHarnesses,
  promptForInstall,
} = require('../lib/prompts');
const { captureStream, makeHome } = require('./helpers');

const adapters = [claude, codex, cursor];

function makePromptApi(selectAnswers, multiAnswer, calls) {
  return {
    async select(options) {
      calls.push({ type: 'select', options });
      return selectAnswers.shift();
    },
    async multiSelect(options) {
      calls.push({ type: 'multiSelect', options });
      return multiAnswer;
    },
  };
}

test('detects Claude, Codex, and Cursor across global and project roots', async (t) => {
  const homeDir = await makeHome(t);
  const cwd = path.join(path.dirname(homeDir), 'project');
  await fs.mkdir(path.join(homeDir, '.claude'), { recursive: true });
  await fs.mkdir(path.join(cwd, '.agents'), { recursive: true });
  await fs.mkdir(path.join(homeDir, '.cursor'), { recursive: true });

  const detected = await detectHarnesses({ adapters, homeDir, cwd });

  assert.deepEqual(detected.map((item) => item.id), ['claude', 'codex', 'cursor']);
  assert.equal(detected[0].displayPath, '~/.claude');
  assert.equal(detected[1].displayPath, path.join(cwd, '.agents'));
  assert.equal(detected[2].displayPath, '~/.cursor');
});

test('deduplicates a harness detected in both global and project roots', async (t) => {
  const homeDir = await makeHome(t);
  const cwd = path.join(path.dirname(homeDir), 'project');
  await fs.mkdir(path.join(homeDir, '.agents'), { recursive: true });
  await fs.mkdir(path.join(cwd, '.agents'), { recursive: true });

  const detected = await detectHarnesses({
    adapters: [claude, codex, codex, cursor],
    homeDir,
    cwd,
  });

  assert.deepEqual(detected, [{ id: 'codex', label: 'Codex', displayPath: '~/.agents' }]);
});

test('detected-only skips customization and defaults to global scope', async () => {
  const calls = [];
  const result = await promptForInstall({
    adapters,
    homeDir: '/tmp/home',
    cwd: '/tmp/project',
    stdin: { isTTY: true },
    stdout: captureStream().stream,
    detected: [
      { id: 'claude', label: 'Claude Code', displayPath: '~/.claude' },
      { id: 'cursor', label: 'Cursor', displayPath: '~/.cursor' },
    ],
    promptApi: makePromptApi(['detected', 'global'], null, calls),
  });

  assert.deepEqual(result, { selectedHarnessIds: ['claude', 'cursor'], scope: 'global' });
  assert.deepEqual(calls.map((call) => call.type), ['select', 'select']);
  assert.equal(calls[0].options.initial, 0);
  assert.equal(calls[1].options.initial, 1);
});

test('customization starts empty and validates an empty selection', async () => {
  const calls = [];
  const result = await promptForInstall({
    adapters,
    homeDir: '/tmp/home',
    cwd: '/tmp/project',
    stdin: { isTTY: true },
    stdout: captureStream().stream,
    detected: [{ id: 'codex', label: 'Codex', displayPath: '~/.agents' }],
    promptApi: makePromptApi(['customize', 'project'], ['claude', 'codex'], calls),
  });

  assert.deepEqual(result, { selectedHarnessIds: ['claude', 'codex'], scope: 'project' });
  const multi = calls.find((call) => call.type === 'multiSelect').options;
  assert.deepEqual(multi.initial, []);
  assert.equal(multi.emptyError, ' ');
  assert.deepEqual(multi.symbols.indicator, { on: '\u25cf', off: '\u25cb' });
  assert.equal(multi.validate([]), 'Select at least one harness.');
  assert.equal(multi.validate(['codex']), true);
});

test('no detections goes directly to customization with all harnesses selected', async () => {
  const calls = [];
  const result = await promptForInstall({
    adapters,
    homeDir: '/tmp/home',
    cwd: '/tmp/project',
    stdin: { isTTY: true },
    stdout: captureStream().stream,
    detected: [],
    promptApi: makePromptApi(['global'], ['claude', 'codex', 'cursor'], calls),
  });

  assert.deepEqual(result.selectedHarnessIds, ['claude', 'codex', 'cursor']);
  assert.deepEqual(calls.map((call) => call.type), ['multiSelect', 'select']);
  assert.deepEqual(calls[0].options.initial, ['claude', 'codex', 'cursor']);
});

test('rejects non-interactive input before detecting or prompting', async () => {
  await assert.rejects(
    promptForInstall({ adapters, stdin: { isTTY: false } }),
    NonInteractiveInstallError,
  );
});
