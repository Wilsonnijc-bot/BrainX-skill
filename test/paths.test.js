'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const codex = require('../adapters/codex');
const claude = require('../adapters/claude');
const { resolveLocations } = require('../lib/paths');

const adapters = [codex, claude];

test('destination resolution supports macOS and Linux homes with spaces', () => {
  for (const home of ['/Users/Test User', '/home/Test User']) {
    const locations = resolveLocations(home, adapters, path.posix);
    assert.equal(locations.destinations.codex, `${home}/.agents/skills`);
    assert.equal(locations.destinations.claude, `${home}/.claude/skills`);
    assert.equal(locations.receiptPath, `${home}/.brainx/receipt.json`);
  }
});

test('destination resolution supports Windows homes with spaces', () => {
  const home = 'C:\\Users\\Test User';
  const locations = resolveLocations(home, adapters, path.win32);
  assert.equal(locations.destinations.codex, 'C:\\Users\\Test User\\.agents\\skills');
  assert.equal(locations.destinations.claude, 'C:\\Users\\Test User\\.claude\\skills');
  assert.equal(locations.receiptPath, 'C:\\Users\\Test User\\.brainx\\receipt.json');
});
