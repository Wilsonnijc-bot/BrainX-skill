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

test('valid commands are dispatched unchanged', async () => {
  let received;
  const status = await runCli(['update'], {
    installer: async (command) => {
      received = command;
      return 7;
    },
  });
  assert.equal(received, 'update');
  assert.equal(status, 7);
});
