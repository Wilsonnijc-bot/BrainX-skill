'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const codex = require('../adapters/codex');
const claude = require('../adapters/claude');
const cursor = require('../adapters/cursor');
const { resolveDestinationRoot, resolveLocations } = require('../lib/paths');

const adapters = [claude, codex, cursor];

test('destination resolution supports macOS and Linux homes with spaces', () => {
  for (const home of ['/Users/Test User', '/home/Test User']) {
    const locations = resolveLocations(home, adapters, path.posix);
    assert.equal(locations.destinations.codex, `${home}/.agents/skills`);
    assert.equal(locations.destinations.claude, `${home}/.claude/skills`);
    assert.equal(locations.destinations.cursor, `${home}/.cursor/skills`);
    assert.equal(locations.receiptPath, `${home}/.brainx/receipt.json`);
  }
});

test('destination resolution supports Windows homes with spaces', () => {
  const home = 'C:\\Users\\Test User';
  const locations = resolveLocations(home, adapters, path.win32);
  assert.equal(locations.destinations.codex, 'C:\\Users\\Test User\\.agents\\skills');
  assert.equal(locations.destinations.claude, 'C:\\Users\\Test User\\.claude\\skills');
  assert.equal(locations.destinations.cursor, 'C:\\Users\\Test User\\.cursor\\skills');
  assert.equal(locations.receiptPath, 'C:\\Users\\Test User\\.brainx\\receipt.json');
});

test('project destinations use the current repository and keep receipt state in home', () => {
  const home = '/Users/Test User';
  const cwd = '/work/BrainX Project';
  const locations = resolveLocations(home, adapters, path.posix, { scope: 'project', cwd });

  assert.equal(locations.destinations.claude, `${cwd}/.claude/skills`);
  assert.equal(locations.destinations.codex, `${cwd}/.agents/skills`);
  assert.equal(locations.destinations.cursor, `${cwd}/.cursor/skills`);
  assert.equal(locations.receiptPath, `${home}/.brainx/receipt.json`);
});

test('destination roots are recovered only from canonical harness paths', () => {
  assert.equal(
    resolveDestinationRoot('/work/project/.cursor/skills', cursor, path.posix),
    '/work/project',
  );
  assert.equal(resolveDestinationRoot('/work/project/skills', cursor, path.posix), null);
});
