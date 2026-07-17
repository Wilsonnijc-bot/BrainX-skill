'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { parseCommand, runCli, USAGE } = require('../lib/cli');
const { captureStream } = require('./helpers');

test('parseCommand accepts only the two public commands', () => {
  assert.equal(parseCommand(['install']), 'install');
  assert.equal(parseCommand(['update']), 'update');
  assert.equal(parseCommand([]), null);
  assert.equal(parseCommand(['skill', 'install']), null);
  assert.equal(parseCommand(['install', 'extra']), null);
  assert.equal(parseCommand(['unknown']), null);
});

test('invalid commands print usage and fail without invoking the installer', async () => {
  const stderr = captureStream();
  let invoked = false;
  const status = await runCli(['skill', 'install'], {
    stderr: stderr.stream,
    installer: async () => {
      invoked = true;
      return 0;
    },
  });

  assert.equal(status, 1);
  assert.equal(invoked, false);
  assert.equal(stderr.text(), `${USAGE}\n`);
});

test('update is dispatched without prompting', async () => {
  let received;
  const status = await runCli(['update'], {
    promptForInstall: async () => {
      throw new Error('update must not prompt');
    },
    installer: async (command) => {
      received = command;
      return 7;
    },
  });
  assert.equal(received, 'update');
  assert.equal(status, 7);
});

test('install selection is passed to the installer', async () => {
  let received;
  const status = await runCli(['install'], {
    homeDir: '/tmp/test-home',
    cwd: '/tmp/test-project',
    promptForInstall: async () => ({
      selectedHarnessIds: ['claude', 'cursor'],
      scope: 'project',
    }),
    installer: async (command, options) => {
      received = { command, options };
      return 0;
    },
  });

  assert.equal(status, 0);
  assert.equal(received.command, 'install');
  assert.deepEqual(received.options.selectedHarnessIds, ['claude', 'cursor']);
  assert.equal(received.options.scope, 'project');
});

test('install fails clearly when stdin is not a terminal', async () => {
  const stderr = captureStream();
  let invoked = false;
  const status = await runCli(['install'], {
    stdin: { isTTY: false },
    stderr: stderr.stream,
    installer: async () => {
      invoked = true;
      return 0;
    },
  });

  assert.equal(status, 1);
  assert.equal(invoked, false);
  assert.equal(
    stderr.text(),
    'Interactive installation requires a terminal.\nRun this command in an interactive shell.\n',
  );
});
