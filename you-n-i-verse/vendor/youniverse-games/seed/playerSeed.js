// PlayerSeedStore — persistent player identity.
// JSON-file persisted to data/players/{playerId}.json.
// The seed is canonical identity: morphs/projections NEVER mutate the seed's
// visual identity or birth/chart state; they only record projection references.

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { safeId, writeJSON } from '../util/storage.js';
import { shortHash } from '../util/hash.js';

export class PlayerSeedStore {
  constructor(dataDir) {
    this.dir = join(dataDir, 'players');
    mkdirSync(this.dir, { recursive: true });
    this._cache = new Map();
  }

  _path(playerId) {
    return join(this.dir, `${safeId(playerId, 'player ID')}.json`);
  }

  exists(playerId) {
    return this._cache.has(playerId) || existsSync(this._path(playerId));
  }

  list() {
    return readdirSync(this.dir).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5));
  }

  /**
   * Create a new PlayerSeed. birthRecord must come from BirthDataStore;
   * visualIdentity must come from photoIngestion. Both are required — a photo
   * that never reaches the seed is a wiring failure by design.
   */
  create({ birthRecord, visualIdentity, visualDNA = null, playerId = null }) {
    if (!birthRecord?.birthDataId) throw new Error('PlayerSeedStore.create: birthRecord required');
    if (!visualIdentity?.fingerprint) throw new Error('PlayerSeedStore.create: visualIdentity required (photo must reach the seed)');
    const id = playerId || `pl-${shortHash(birthRecord.birthDataId + ':' + visualIdentity.fingerprint, 10)}`;
    if (this.exists(id)) throw new Error(`PlayerSeedStore.create: player "${id}" already exists`);

    const seed = {
      playerId: id,
      name: birthRecord.name,
      createdAt: new Date().toISOString(),
      birthDataId: birthRecord.birthDataId, // canonical reference into birth store
      sourceImage: visualIdentity.sourceImageRef,
      visualIdentityFeatures: visualIdentity, // canonical — never mutated by morphs
      // PHASE 1: extracted facial identity anchors + the continuity baseline
      // morphs are measured against. Canonical, alongside visualIdentityFeatures.
      visualDNA: visualDNA ?? { anchors: null, continuityBaseline: null, reason: 'not computed at intake' },
      humanDesignState: structuredClone(birthRecord.humanDesign), // hydrated canonical reference
      ichingState: structuredClone(birthRecord.iching),
      persistentProgression: { level: 1, xp: 0, insights: 0, milestones: [] },
      gameHistory: [],
      earnedArtifacts: [],
      relationships: [],
      transformations: [],
      currentWorld: null,
      currentEmbodiment: null, // projection reference only
    };
    this._save(seed);
    this._cache.set(id, seed);
    return structuredClone(seed);
  }

  load(playerId) {
    if (this._cache.has(playerId)) return structuredClone(this._cache.get(playerId));
    const path = this._path(playerId);
    if (!existsSync(path)) throw new Error(`PlayerSeedStore.load: no seed for "${playerId}"`);
    const seed = JSON.parse(readFileSync(path, 'utf8'));
    this._cache.set(playerId, seed);
    return structuredClone(seed);
  }

  /**
   * Update allowed (non-canonical) fields. Attempts to mutate canonical
   * identity (visualIdentityFeatures, humanDesignState, ichingState, birthDataId)
   * through update() throw — morphs must go through recordTransformation().
   */
  update(playerId, patch) {
    const forbidden = ['visualIdentityFeatures', 'visualDNA', 'humanDesignState', 'ichingState', 'birthDataId', 'playerId', 'sourceImage'];
    for (const k of Object.keys(patch)) {
      if (forbidden.includes(k)) {
        throw new Error(`PlayerSeedStore.update: "${k}" is canonical identity and cannot be mutated by projections`);
      }
    }
    const seed = this._cache.get(playerId) ?? JSON.parse(readFileSync(this._path(playerId), 'utf8'));
    Object.assign(seed, structuredClone(patch));
    this._save(seed);
    this._cache.set(playerId, seed);
    return structuredClone(seed);
  }

  recordTransformation(playerId, morphState) {
    const seed = this._cache.get(playerId) ?? JSON.parse(readFileSync(this._path(playerId), 'utf8'));
    seed.transformations.push({
      embodimentId: morphState.embodimentId,
      gameId: morphState.gameId,
      sceneId: morphState.sceneId ?? null,
      kind: morphState.kind,
      species: morphState.transformationRules?.species ?? null,
      biome: morphState.transformationRules?.biome ?? null,
      stateHash: morphState.stateHash ?? null,
      morphState: structuredClone(morphState),
      reversible: morphState.reversible,
      at: new Date().toISOString(),
    });
    seed.currentEmbodiment = {
      embodimentId: morphState.embodimentId,
      gameId: morphState.gameId,
      kind: morphState.kind,
      reversible: morphState.reversible,
    };
    this._save(seed);
    this._cache.set(playerId, seed);
    return structuredClone(seed);
  }

  addGameHistory(playerId, entry) {
    const seed = this.load(playerId);
    seed.gameHistory.push(entry);
    return this.update(playerId, { gameHistory: seed.gameHistory });
  }

  _save(seed) {
    writeJSON(this._path(seed.playerId), seed);
  }
}
