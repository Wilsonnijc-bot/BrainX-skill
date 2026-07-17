'use strict';

const path = require('node:path');

function resolveLocations(homeDir, adapters, pathApi = path) {
  if (typeof homeDir !== 'string' || !pathApi.isAbsolute(homeDir)) {
    throw new Error(`Home directory must be an absolute path: ${homeDir}`);
  }

  const destinations = {};
  for (const adapter of adapters) {
    destinations[adapter.id] = pathApi.resolve(homeDir, ...adapter.homePath);
  }

  return {
    homeDir: pathApi.resolve(homeDir),
    stateDir: pathApi.resolve(homeDir, '.brainx'),
    receiptPath: pathApi.resolve(homeDir, '.brainx', 'receipt.json'),
    destinations,
  };
}

function samePath(left, right, pathApi = path) {
  const normalizedLeft = pathApi.normalize(pathApi.resolve(left));
  const normalizedRight = pathApi.normalize(pathApi.resolve(right));
  if (pathApi === path.win32) {
    return normalizedLeft.toLowerCase() === normalizedRight.toLowerCase();
  }
  return normalizedLeft === normalizedRight;
}

module.exports = {
  resolveLocations,
  samePath,
};
