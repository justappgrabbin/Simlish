// GameState — per-world game-specific state, JSON persisted to
// data/gamesaves/{playerId}/{worldName}.json. Strictly separate from the
// canonical PlayerSeed: story position, local inventory, current quest,
// current morph, local relationships, local world state live here only.

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { safeId, writeJSON } from '../util/storage.js';

export class GameStateStore {
  constructor(dataDir) {
    this.root = join(dataDir, 'gamesaves');
    mkdirSync(this.root, { recursive: true });
  }

  _path(playerId, worldName) {
    return join(this.root, safeId(playerId), `${safeId(worldName)}.json`);
  }

  exists(playerId, worldName) {
    return existsSync(this._path(playerId, worldName));
  }

  createDefault(playerId, worldName) {
    return {
      playerId,
      worldName,
      storyPosition: { sceneIndex: 0, chapter: 1 },
      localInventory: [],
      currentQuest: null,
      currentMorph: null, // projection reference (embodimentId), not identity
      localRelationships: [],
      localWorldState: {},
      ichingNextState: null,
      tick: 0,
    };
  }

  load(playerId, worldName) {
    if (!this.exists(playerId, worldName)) return this.createDefault(playerId, worldName);
    return JSON.parse(readFileSync(this._path(playerId, worldName), 'utf8'));
  }

  save(state) {
    if (!state?.playerId || !state?.worldName) throw new Error('GameStateStore.save: state must have playerId and worldName');
    const dir = join(this.root, state.playerId);
    mkdirSync(dir, { recursive: true });
    writeJSON(this._path(state.playerId, state.worldName), state);
    return this._path(state.playerId, state.worldName);
  }
}
