'use strict';

const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');

async function makeTemporaryDirectory(t, label = 'brainx test ') {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), label));
  t.after(async () => {
    await fs.rm(directory, { recursive: true, force: true });
  });
  return directory;
}

async function makeBundle(t, options = {}) {
  const version = options.version || '1.0.0';
  const skills = options.skills || ['alpha', 'beta'];
  const contentTag = options.contentTag || version;
  const workspace = await makeTemporaryDirectory(t, 'brainx bundle ');
  const packageRoot = path.join(workspace, `package ${version}`);
  await fs.mkdir(path.join(packageRoot, 'skills'), { recursive: true });
  await fs.writeFile(path.join(packageRoot, 'package.json'), `${JSON.stringify({
    name: 'brainx-skill',
    version,
  }, null, 2)}\n`);
  await fs.writeFile(path.join(packageRoot, 'manifest.json'), `${JSON.stringify({
    schemaVersion: 1,
    skills,
  }, null, 2)}\n`);

  for (const skill of skills) {
    const skillRoot = path.join(packageRoot, 'skills', skill);
    await fs.mkdir(path.join(skillRoot, 'references'), { recursive: true });
    await fs.writeFile(
      path.join(skillRoot, 'SKILL.md'),
      `---\nname: ${skill}\ndescription: Test skill\n---\n\n# ${skill}\n\n${contentTag}\n`,
    );
    await fs.writeFile(path.join(skillRoot, 'references', 'reference.txt'), `${contentTag}\n`);
  }
  return packageRoot;
}

async function makeHome(t) {
  const workspace = await makeTemporaryDirectory(t, 'brainx home ');
  const homeDir = path.join(workspace, 'User Home With Spaces');
  await fs.mkdir(homeDir);
  return homeDir;
}

function captureStream() {
  let value = '';
  return {
    stream: {
      write(chunk) {
        value += String(chunk);
        return true;
      },
    },
    text() {
      return value;
    },
  };
}

function sequentialUuid(prefix = 'test') {
  let counter = 0;
  return () => `${prefix}-${++counter}`;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

module.exports = {
  captureStream,
  makeBundle,
  makeHome,
  makeTemporaryDirectory,
  readJson,
  sequentialUuid,
};
