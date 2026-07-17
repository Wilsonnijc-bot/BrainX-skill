'use strict';

const { runInstaller } = require('./installer');

const USAGE = `Usage:
  npx brainx-skills install
  npx brainx-skills update`;

function parseCommand(args) {
  if (args.length !== 1 || (args[0] !== 'install' && args[0] !== 'update')) {
    return null;
  }
  return args[0];
}

async function runCli(args, options = {}) {
  const stderr = options.stderr || process.stderr;
  const command = parseCommand(args);
  if (!command) {
    stderr.write(`${USAGE}\n`);
    return 1;
  }
  const installer = options.installer || runInstaller;
  return installer(command, options);
}

module.exports = {
  USAGE,
  parseCommand,
  runCli,
};
