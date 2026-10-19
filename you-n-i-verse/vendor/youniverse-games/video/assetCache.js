// Asset cache with prompt-hash dedup: identical scene DNA never regenerates.

import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { shortHash } from '../util/hash.js';

export class AssetCache {
  constructor(dir) {
    this.dir = dir;
    mkdirSync(this.dir, { recursive: true });
    this._indexPath = join(this.dir, 'index.json');
    this._index = existsSync(this._indexPath) ? JSON.parse(readFileSync(this._indexPath, 'utf8')) : {};
  }

  /** Content hash of a scene DNA payload (excluding volatile fields). */
  hashScene(scenePayload) {
    const { hash: _h, ...rest } = scenePayload;
    return shortHash(JSON.stringify(rest), 24);
  }

  has(hash) {
    const entry = this._index[hash];
    return !!entry && existsSync(entry.path);
  }

  get(hash) {
    return this.has(hash) ? { ...this._index[hash], cached: true } : null;
  }

  put(hash, asset, sceneDNA) {
    this._index[hash] = { ...asset, dnaPath: asset.dnaPath ?? null, createdAt: new Date().toISOString() };
    writeFileSync(this._indexPath, JSON.stringify(this._index, null, 2));
    return this._index[hash];
  }

  size() { return Object.keys(this._index).length; }
}
