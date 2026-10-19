// WorldAdapter — the seam between the runtime and game worlds.
// See world/README.md for the world-module interface contract.

export class WorldAdapter {
  /**
   * @param {object} deps - {seedStore, gameStateStore, morphEngine, renderer}
   */
  constructor(deps) {
    for (const k of ['seedStore', 'gameStateStore', 'morphEngine', 'renderer']) {
      if (!deps[k]) throw new Error(`WorldAdapter: missing dependency "${k}"`);
    }
    this.deps = deps;
    this.seedStore = deps.seedStore;
    this.gameStateStore = deps.gameStateStore;
    this.worlds = new Map();
    this.activeSessions = new Map(); // playerId -> {worldName, instance}
  }

  registerWorld(name, moduleFactory) {
    if (typeof moduleFactory !== 'function' && typeof moduleFactory?.create !== 'function') {
      throw new Error('WorldAdapter.registerWorld: module must be a factory function or {create()}');
    }
    this.worlds.set(name, moduleFactory);
  }

  listWorlds() {
    return [...this.worlds.keys()];
  }

  /**
   * Enter a world: loads game state, creates the world instance, calls
   * world.enter(seed, adapter). Returns the world's entry scene/state.
   */
  enterWorld(playerId, worldName) {
    const factory = this.worlds.get(worldName);
    if (!factory) throw new Error(`WorldAdapter: world "${worldName}" not registered (have: ${this.listWorlds().join(', ')})`);
    const seed = this.seedStore.load(playerId);
    const gameState = this.gameStateStore.load(playerId, worldName);
    const instance = typeof factory === 'function' ? factory() : factory.create();
    for (const m of ['enter', 'handleChoice', 'getState', 'save', 'load']) {
      if (typeof instance[m] !== 'function') throw new Error(`WorldAdapter: world "${worldName}" missing method ${m}()`);
    }
    const ctx = { seed, gameState, adapter: this, morphEngine: this.deps.morphEngine, renderer: this.deps.renderer };
    const entry = instance.enter(ctx);
    this.activeSessions.set(playerId, { worldName, instance, ctx });
    this.seedStore.update(playerId, { currentWorld: worldName });
    const remember = scene => { gameState.lastScene = structuredClone(scene); return scene; };
    return entry && typeof entry.then === 'function' ? entry.then(remember) : remember(entry);
  }

  handleChoice(playerId, choice) {
    const session = this.activeSessions.get(playerId);
    if (!session) throw new Error(`WorldAdapter: player "${playerId}" is not in any world`);
    const result = session.instance.handleChoice(choice);
    const remember = scene => { session.ctx.gameState.lastScene = structuredClone(scene); return scene; };
    // Refresh canonical seed reference to current embodiment after consequences.
    return result && typeof result.then === 'function' ? result.then(remember) : remember(result);
  }

  getState(playerId) {
    const session = this.activeSessions.get(playerId);
    if (!session) throw new Error(`WorldAdapter: player "${playerId}" is not in any world`);
    return session.instance.getState();
  }

  saveWorld(playerId) {
    const session = this.activeSessions.get(playerId);
    if (!session) throw new Error(`WorldAdapter: player "${playerId}" is not in any world`);
    return session.instance.save();
  }

  exitWorld(playerId) {
    const session = this.activeSessions.get(playerId);
    if (!session) throw new Error(`WorldAdapter: player "${playerId}" is not in any world`);
    session.instance.save();
    this.activeSessions.delete(playerId);
    this.seedStore.update(playerId, { currentWorld: null });
    return true;
  }

  morphAndRender(seed, ctx) {
    const morph = this.deps.morphEngine.morph(seed, ctx);
    const asset = this.deps.renderer.render(morph);
    return { morph, asset };
  }
}
