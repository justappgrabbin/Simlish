// world/reactiveLoop.js — UPGRADE_PLAN §5. A real per-character simulation
// loop. Each reactive character owns: synthetic visualDNA anchors (distinct,
// deterministic), needs/drives/tensions, a relationship map (player + other
// NPCs), perception state, form constraints, environment response rules, an
// OntologicalAddress and a history.
//
// Tick: perceive -> resolve (shared ichingResolver) -> react (action from
// drives+tensions+relationships; actions mutate world state, tensions and
// relationships — including OTHER characters, producing NPC->NPC chains) ->
// maybeMorph (tension threshold triggers the klein operator chain; embodiment
// rendered via embodimentRenderer; identity continuity asserted) -> record
// (history + address update). Deterministic seeded RNG per tick.
//
// State is plain JSON (getState/fromState) so it persists inside game saves.

import { sha256Hex, seededRng, deriveFloats, pickSeeded } from '../util/hash.js';
import { resolveIChing } from '../iching/ichingResolver.js';
import { runKleinChain } from '../morph/kleinOperators.js';
import { identityContinuity, CONTINUITY_THRESHOLD } from '../identity/continuity.js';
import { createAddress, recordEvent } from '../state/ontologicalAddress.js';

const r3 = (v) => +v.toFixed(3);
const clamp01 = (v) => Math.max(0, Math.min(1, v));

// Distinct synthetic IdentityAnchors per NPC — same shape as visualDNA output,
// deterministic from the NPC id. NPCs never read the player's photo anchors.
export function synthesizeAnchors(npcId, purpose) {
  const floats = deriveFloats(`npc-anchors:${npcId}:${purpose}`, 16);
  const skinBase = 120 + floats[0] * 110;          // warm-ish skin range
  const anchors = {
    version: 1,
    faceBBox: { x: 20, y: 18, w: 70 + Math.round(floats[1] * 20), h: 88 + Math.round(floats[2] * 24) },
    faceRatio: r3(0.62 + floats[3] * 0.28),
    centroid: { x: r3(0.42 + floats[4] * 0.16), y: r3(0.48 + floats[5] * 0.12) },
    eyeSpacingRatio: r3(0.3 + floats[6] * 0.24),
    eyeY: r3(0.34 + floats[7] * 0.14),
    mouthY: r3(0.66 + floats[8] * 0.18),
    expression: {
      mouthDarkness: r3(0.25 + floats[9] * 0.5),
      mouthCurve: r3((floats[10] - 0.5) * 0.5),
      browContrast: r3(0.15 + floats[11] * 0.5),
    },
    silhouetteHash: sha256Hex(`npc-silhouette:${npcId}`).slice(0, 64),
    skinTone: [Math.round(skinBase), Math.round(skinBase * 0.72), Math.round(skinBase * 0.55)],
    hairTone: [Math.round(20 + floats[12] * 70), Math.round(16 + floats[13] * 50), Math.round(14 + floats[14] * 40)],
    anchorsHash: null,
  };
  anchors.faceRatio = r3(anchors.faceBBox.w / anchors.faceBBox.h);
  anchors.anchorsHash = sha256Hex({ ...anchors, anchorsHash: undefined });
  return anchors;
}

// Drive -> action table. Action selection scores each action by the acting
// character's drives, tensions and relationships; effects are concrete:
// tension/disposition shifts, world-state notes, cross-NPC disturbance.
const ACTIONS = {
  'greet-player':     { drive: 'approach', need: 'belonging', target: 'player', effects: { relPlayer: +0.15, tensionPlayer: -0.06, disturbance: 0.05 }, label: 'greets the player warmly' },
  'aid-player':       { drive: 'nurture',  need: 'purpose',   target: 'player', effects: { relPlayer: +0.2,  tensionPlayer: -0.1,  disturbance: 0.03, worldNote: 'an ally clears the path ahead' }, label: 'aids the player' },
  'challenge-player': { drive: 'guard',    need: 'safety',    target: 'player', effects: { relPlayer: -0.2,  tensionPlayer: +0.22, disturbance: 0.3 }, label: 'challenges the player' },
  'watch-player':     { drive: 'guard',    need: 'safety',    target: 'player', effects: { relPlayer: 0,     tensionPlayer: +0.05, disturbance: 0.02 }, label: 'watches the player narrowly' },
  'console-ally':     { drive: 'nurture',  need: 'belonging', target: 'npc',    effects: { relNpc: +0.25,    tensionNpc: -0.15,    disturbance: -0.05 }, label: 'consoles a shaken companion' },
  'rebuke-npc':       { drive: 'guard',    need: 'belonging', target: 'npc',    effects: { relNpc: -0.3,     tensionNpc: +0.25,    disturbance: 0.2 }, label: 'rebukes another villager' },
  'withdraw':         { drive: 'guard',    need: 'safety',    target: 'self',   effects: { tensionPlayer: -0.05, disturbance: -0.02 }, label: 'withdraws from the scene' },
};
const ACTION_IDS = Object.keys(ACTIONS);

// Drive signature -> species used when the character morphs under pressure.
const DRIVE_SPECIES = { guard: 'feline', nurture: 'terrane', approach: 'avian' };

export class ReactiveLoop {
  /**
   * @param {object} deps - {worldId, renderer? (EmbodimentRenderer), morphThreshold?,
   *   minContinuity?, playerRef? ({entityId, identityRef})}
   */
  constructor(deps) {
    if (!deps?.worldId) throw new Error('ReactiveLoop: deps.worldId required');
    this.worldId = deps.worldId;
    this.renderer = deps.renderer ?? null; // NPC embodiments rendered when present
    this.morphThreshold = deps.morphThreshold ?? 0.75;
    this.minContinuity = deps.minContinuity ?? CONTINUITY_THRESHOLD;
    this.playerRef = deps.playerRef ?? null;
    this.characters = new Map(); // id -> character state (plain JSON)
    this.log = [];               // tick log (plain JSON)
  }

  /** Register a persistent reactive NPC. Idempotent per id. */
  registerCharacter({ id, purpose, species = 'hybrid', biome = 'forest', lineageMarkers = [] }) {
    if (this.characters.has(id)) return this.characters.get(id);
    const anchors = synthesizeAnchors(id, purpose);
    const floats = deriveFloats(`npc-drives:${id}`, 6);
    const char = {
      id,
      role: 'npc',
      purpose,
      form: { species, biome },          // current form
      baseForm: { species, biome },
      visualDNA: { anchors, continuityBaseline: structuredClone(anchors) },
      needs: { safety: r3(0.4 + floats[0] * 0.3), belonging: r3(0.4 + floats[1] * 0.3), purpose: r3(0.4 + floats[2] * 0.3) },
      drives: { approach: r3(0.3 + floats[3] * 0.4), guard: r3(0.3 + floats[4] * 0.4), nurture: r3(0.3 + floats[5] * 0.4) },
      tensions: { player: 0.1 },         // npc -> player + other npc ids
      relationshipMap: { player: 0 },    // disposition toward player + npcs
      perceptionState: { lastPlayerAction: null, lastWorldNote: null, observed: [] },
      formConstraints: { minContinuity: this.minContinuity, morphThreshold: this.morphThreshold },
      environmentResponseRules: [
        { when: 'pressure contains conflict', shift: { 'drives.guard': +0.1 } },
        { when: 'pressure contains increase', shift: { 'drives.approach': +0.1 } },
      ],
      address: createAddress({ entityId: `npc:${id}`, identityRef: anchors.anchorsHash, worldId: this.worldId, purpose, lineageMarkers }),
      history: [],
      morphs: [],
      lastAction: null,
    };
    // other NPCs enter the relationship map / tension map
    for (const other of this.characters.values()) {
      char.tensions[other.id] = 0.1;
      char.relationshipMap[other.id] = 0;
      other.tensions[id] = 0.1;
      other.relationshipMap[id] = 0;
    }
    this.characters.set(id, char);
    return char;
  }

  /**
   * Advance every character one tick.
   * @param {object} input - {worldState, playerAction?, tick}
   *   playerAction: {id, target?} — what the player just did (observable).
   * @returns {object} tick result {actions, chainEvents, morphs, worldNotes}
   */
  tick({ worldState, playerAction = null, tick = 0 }) {
    const rng = seededRng(`reactive:${this.worldId}:${tick}:${playerAction?.id ?? 'none'}`);
    const result = { tick, actions: [], chainEvents: [], morphs: [], worldNotes: [] };

    // Resolution through the shared I Ching core — the world's reading shapes
    // every character this tick (pressure feeds environmentResponseRules).
    const hex = worldState.activeHexagram ?? 1;
    const line = worldState.activeLine ?? 1;
    const resolution = resolveIChing({
      playerState: { playerId: this.playerRef?.entityId ?? 'player' },
      worldState,
      relationshipState: { npcs: [...this.characters.values()].map((c) => ({ id: c.id })) },
      activeHexagram: hex,
      activeLine: line,
      previousState: { hexagram: hex, sceneIndex: worldState.sceneIndex ?? 0 },
      event: 'reactive-tick',
      choice: playerAction?.id ?? null,
      temporalState: { tick },
    });

    for (const char of this.characters.values()) {
      // 1. PERCEIVE -----------------------------------------------------------
      char.perceptionState.lastPlayerAction = playerAction?.id ?? null;
      char.perceptionState.pressure = resolution.narrativePressure;
      char.perceptionState.observed = [...this.characters.values()]
        .filter((o) => o.id !== char.id && o.lastAction)
        .map((o) => ({ id: o.id, action: o.lastAction.action, disturbance: o.lastAction.disturbance }));

      // environment response rules: resolution pressure shifts drives
      for (const rule of char.environmentResponseRules) {
        const [kind, key] = rule.when.split(' contains ');
        if (kind === 'pressure' && resolution.narrativePressure.includes(key)) {
          for (const [path, delta] of Object.entries(rule.shift)) {
            const [, field] = path.split('.');
            char.drives[field] = r3(clamp01(char.drives[field] + delta));
          }
        }
      }
      // player action is perceived directly: confrontational choices raise
      // tension, generous ones lower it (observable player -> NPC coupling).
      const playerAggro = { 'challenge-player': 0.2, confront: 0.25, depart: 0.05, approach: -0.05, offer: -0.12, observe: 0 };
      const direct = playerAggro[playerAction?.id] ?? 0;
      char.tensions.player = r3(clamp01(char.tensions.player + direct));
      char.relationshipMap.player = r3(char.relationshipMap.player - direct * 0.5);
      if (playerAction?.target === char.id) {
        char.relationshipMap.player = r3(char.relationshipMap.player + 0.15);
        char.tensions.player = r3(clamp01(char.tensions.player - 0.08));
      }

      // NPC -> NPC perception: observed disturbance from others raises tension
      // toward the disturber; nurturing characters also grow protective of the
      // player when someone else disturbs the scene (the reaction chain).
      for (const obs of char.perceptionState.observed) {
        if (obs.disturbance > 0.1) {
          char.tensions[obs.id] = r3(clamp01((char.tensions[obs.id] ?? 0) + obs.disturbance * char.drives.guard));
          char.relationshipMap[obs.id] = r3((char.relationshipMap[obs.id] ?? 0) - obs.disturbance * 0.5);
          if (char.drives.nurture > char.drives.guard) {
            char.relationshipMap.player = r3(char.relationshipMap.player + obs.disturbance * 0.3);
            result.chainEvents.push({
              tick, cause: { npc: obs.id, action: obs.action },
              effect: { npc: char.id, relationshipShiftToward: obs.id, protectPlayer: true },
              description: `${char.id} saw ${obs.id} ${obs.action}; tension toward ${obs.id} rose, protectiveness toward player grew`,
            });
          } else {
            result.chainEvents.push({
              tick, cause: { npc: obs.id, action: obs.action },
              effect: { npc: char.id, relationshipShiftToward: obs.id, protectPlayer: false },
              description: `${char.id} saw ${obs.id} ${obs.action}; tension toward ${obs.id} rose`,
            });
          }
        }
      }

      // 3. REACT — action selection from drives + tensions + relationships ---
      const action = this._selectAction(char, rng);
      this._applyAction(char, action, result);

      // 4. MAYBE MORPH --------------------------------------------------------
      const tension = Math.max(char.tensions.player, ...Object.entries(char.tensions).filter(([k]) => k !== 'player').map(([, v]) => v));
      const targetSpecies = DRIVE_SPECIES[this._dominantDrive(char)] ?? 'hybrid';
      // morph when threshold crossed AND the pressure demands a NEW form
      // (or tension is at maximum — re-fusion under extreme pressure)
      if (tension >= char.formConstraints.morphThreshold && (char.form.species !== targetSpecies || tension >= 0.98)) {
        const morphRecord = this._morph(char, tension, tick);
        result.morphs.push(morphRecord);
      }

      // 5. RECORD -------------------------------------------------------------
      const entry = {
        tick, action: action.id, target: action.targetId,
        tensionPlayer: char.tensions.player,
        dispositionPlayer: char.relationshipMap.player,
        dominantDrive: this._dominantDrive(char),
      };
      char.history.push(entry);
      recordEvent(char.address, { tick, event: `tick:${action.id}`, detail: { target: action.targetId, tensionPlayer: char.tensions.player } });
    }

    this.log.push({ tick, pressure: resolution.narrativePressure, actions: result.actions.map((a) => `${a.npc}:${a.action}`), chains: result.chainEvents.length, morphs: result.morphs.length });
    return result;
  }

  _dominantDrive(char) {
    return Object.entries(char.drives).sort((x, y) => y[1] - x[1])[0][0];
  }

  _selectAction(char, rng) {
    const others = [...this.characters.values()].filter((o) => o.id !== char.id);
    let best = null;
    for (const id of ACTION_IDS) {
      const def = ACTIONS[id];
      let score = char.drives[def.drive] * 0.6 + rng() * 0.15;
      // tensions push toward the matching posture
      if (def.target === 'player') score += char.tensions.player * (def.drive === 'guard' ? 0.5 : -0.2);
      score += char.relationshipMap.player * (def.effects.relPlayer > 0 ? 0.4 : def.effects.relPlayer < 0 ? -0.4 : 0);
      if (def.target === 'npc') {
        if (others.length === 0) continue;
        // console the most tense ally; rebuke the most disturbing one
        const worst = others.reduce((a, b) => ((char.tensions[b.id] ?? 0) > (char.tensions[a.id] ?? 0) ? b : a));
        const tensionTo = char.tensions[worst.id] ?? 0;
        score += id === 'rebuke-npc' ? tensionTo * 0.6 : (1 - tensionTo) * char.drives.nurture * 0.3;
        var npcTarget = worst.id;
      }
      if (id === 'withdraw' && char.tensions.player > 0.5 && char.drives.guard > 0.6) score += 0.3;
      if (!best || score > best.score) best = { id, def, score, npcTarget: def.target === 'npc' ? npcTarget : null };
    }
    const targetId = best.def.target === 'player' ? 'player' : best.def.target === 'npc' ? best.npcTarget : char.id;
    return { id: best.id, def: best.def, targetId };
  }

  _applyAction(char, action, result) {
    const e = action.def.effects;
    if (e.relPlayer) char.relationshipMap.player = r3(char.relationshipMap.player + e.relPlayer);
    if (e.tensionPlayer) char.tensions.player = r3(clamp01(char.tensions.player + e.tensionPlayer));
    if (action.targetId && action.targetId !== 'player' && action.targetId !== char.id) {
      const other = this.characters.get(action.targetId);
      if (other) {
        if (e.relNpc) {
          char.relationshipMap[other.id] = r3((char.relationshipMap[other.id] ?? 0) + e.relNpc);
          other.relationshipMap[char.id] = r3((other.relationshipMap[char.id] ?? 0) + e.relNpc * 0.7);
        }
        if (e.tensionNpc) {
          char.tensions[other.id] = r3(clamp01((char.tensions[other.id] ?? 0) + e.tensionNpc));
          other.tensions[char.id] = r3(clamp01((other.tensions[char.id] ?? 0) + Math.abs(e.tensionNpc) * 0.6));
        }
      }
    }
    // needs drift toward their drives (the character is alive between ticks)
    for (const [need, v] of Object.entries(char.needs)) {
      char.needs[need] = r3(clamp01(v + (char.drives[action.def.drive] - 0.5) * 0.05 + (need === action.def.need ? 0.03 : -0.01)));
    }
    char.lastAction = { action: action.id, target: action.targetId, disturbance: e.disturbance ?? 0, label: action.def.label };
    result.actions.push({ npc: char.id, action: action.id, target: action.targetId, label: `${char.id} ${action.def.label}`, disturbance: e.disturbance ?? 0 });
    if (e.worldNote) result.worldNotes.push(e.worldNote);
  }

  _morph(char, tension, tick) {
    const dominant = this._dominantDrive(char);
    const species = DRIVE_SPECIES[dominant] ?? 'hybrid';
    const spec = runKleinChain(char.visualDNA, { species, biome: char.form.biome, lineageMarkers: char.address.lineageMarkers });
    const cont = identityContinuity(char.visualDNA.continuityBaseline, spec.anchorsAfter);
    if (cont.score < char.formConstraints.minContinuity) {
      throw new Error(`ReactiveLoop: NPC "${char.id}" morph broke identity continuity (${cont.score} < ${char.formConstraints.minContinuity})`);
    }
    let render = null;
    if (this.renderer) {
      render = this.renderer.render(spec, { playerId: char.id, embodimentId: `npc-${tick}` });
    }
    char.form = { species, biome: char.form.biome };
    const record = {
      npc: char.id, tick, trigger: `tension ${tension} >= ${char.formConstraints.morphThreshold}`,
      dominantDrive: dominant, fromSpecies: char.baseForm.species, toSpecies: species,
      continuity: cont.score, recognizable: cont.recognizable,
      specHash: spec.specHash.slice(0, 16), renderPath: render?.path ?? null,
    };
    char.morphs.push(record);
    // morphing vents tension — the character has changed to cope
    char.tensions.player = r3(clamp01(char.tensions.player - 0.3));
    recordEvent(char.address, { tick, event: `morph:${species}`, detail: { continuity: cont.score } });
    return record;
  }

  // --- persistence (plain JSON) ----------------------------------------------
  getState() {
    return {
      worldId: this.worldId,
      morphThreshold: this.morphThreshold,
      minContinuity: this.minContinuity,
      playerRef: this.playerRef,
      characters: [...this.characters.values()].map((c) => structuredClone(c)),
      log: structuredClone(this.log),
    };
  }

  /** Restore from a saved state (fresh instance, e.g. after save/exit/reload). */
  static fromState(saved, deps = {}) {
    if (!saved?.worldId || !Array.isArray(saved.characters)) {
      throw new Error('ReactiveLoop.fromState: saved state with worldId and characters required');
    }
    const loop = new ReactiveLoop({ ...deps, worldId: saved.worldId, morphThreshold: saved.morphThreshold, minContinuity: saved.minContinuity, playerRef: saved.playerRef });
    for (const c of saved.characters) loop.characters.set(c.id, structuredClone(c));
    loop.log = structuredClone(saved.log ?? []);
    return loop;
  }
}

export { ACTIONS as REACTIVE_ACTIONS, pickSeeded };
