// Journey State Manager (spec §7.1 / §7.2) for The Oracle's Thread.
// Lives INSIDE gameState.localWorldState.oraclesThread so persistence flows
// through the existing state/gameState.js mechanism — no parallel store.
//
// Schema (§7.1):
//   birth_signature: { pillars: {year,month,day,hour}, elements, birthDataId }
//   current_journey: { primary_hexagram, resulting_hexagram, changing_lines,
//                      situation_history, accumulated_resonance,
//                      current_elemental_state, destiny_vector (512),
//                      scenes_visited, rewards, phase }
//   session_state:   { scene, available_entities, situation_intent_vector,
//                      intent_entity, choice_deadline, last_band }
//
// State transitions (§7.2):
//   ENTERING -> DIVINING -> SCENE_ACTIVE -> RESOLVING -> SCENE_ACTIVE (loop)
//   Any active phase -> EXITING (via save/exit). Distortion never exits: it
//   routes to shadow content and returns to SCENE_ACTIVE.

export const PHASES = ['ENTERING', 'DIVINING', 'SCENE_ACTIVE', 'RESOLVING', 'EXITING'];

export class JourneyStateManager {
  /** @param {object} gameState - the world gameState from GameStateStore */
  constructor(gameState) {
    this.gameState = gameState;
  }

  exists() {
    return !!this.gameState.localWorldState?.oraclesThread;
  }

  /** Initialize journey state (§7.1). Throws if already initialized. */
  initialize({ birthDataId, pillars, elements, hexagrams, destinyVector }) {
    if (this.exists()) throw new Error('JourneyStateManager: already initialized');
    const journey = {
      birth_signature: {
        birthDataId,
        pillars: {
          year: pillars.year.text, month: pillars.month.text,
          day: pillars.day.text, hour: pillars.hour.text,
        },
        pillarDetail: pillars,
        elements,
      },
      current_journey: {
        phase: 'ENTERING',
        primary_hexagram: hexagrams.primary,
        resulting_hexagram: hexagrams.resulting,
        changing_lines: hexagrams.changingLines,
        situation_history: [],
        accumulated_resonance: 0,
        current_elemental_state: { ...elements.counts },
        destiny_vector: destinyVector,
        scenes_visited: 0,
        rewards: [],
      },
      session_state: {
        scene: null,
        available_entities: [],
        situation_intent_vector: null,
        intent_entity: null,
        choice_deadline: null,
        last_band: null,
      },
    };
    this.gameState.localWorldState = { ...this.gameState.localWorldState, oraclesThread: journey };
    return journey;
  }

  get() {
    const j = this.gameState.localWorldState?.oraclesThread;
    if (!j) throw new Error('JourneyStateManager: journey not initialized');
    return j;
  }

  /** §7.2 transition guard. */
  transition(toPhase) {
    if (!PHASES.includes(toPhase)) throw new Error(`JourneyStateManager: unknown phase "${toPhase}"`);
    const j = this.get();
    const from = j.current_journey.phase;
    const legal = {
      ENTERING: ['DIVINING'],
      DIVINING: ['SCENE_ACTIVE'],
      SCENE_ACTIVE: ['RESOLVING', 'EXITING'],
      RESOLVING: ['SCENE_ACTIVE', 'EXITING'],
      EXITING: [],
    };
    if (!legal[from]?.includes(toPhase)) {
      throw new Error(`JourneyStateManager: illegal transition ${from} -> ${toPhase}`);
    }
    j.current_journey.phase = toPhase;
    return { from, to: toPhase };
  }

  /** Begin a scene: set available entities, intent vector, deadline (§7.1 session_state). */
  beginScene({ entities, intentEntity, intentVector, durationMs = 90000, sceneHash, now = new Date() }) {
    const j = this.get();
    j.session_state.scene = sceneHash;
    j.session_state.available_entities = entities;
    j.session_state.intent_entity = intentEntity;
    j.session_state.situation_intent_vector = intentVector;
    j.session_state.choice_deadline = new Date(now.getTime() + durationMs).toISOString();
    j.current_journey.scenes_visited += 1;
    return j.session_state;
  }

  /** Record a resolved interaction into journey state. */
  recordOutcome({ score, band, effects, interactionText, newDestinyVector, elementDelta = null }) {
    const j = this.get();
    j.current_journey.accumulated_resonance = +(j.current_journey.accumulated_resonance + score).toFixed(4);
    j.current_journey.destiny_vector = newDestinyVector;
    j.current_journey.situation_history.push({ at: new Date().toISOString(), interaction: interactionText, score, band });
    j.session_state.last_band = band;
    if (elementDelta) {
      for (const [el, d] of Object.entries(elementDelta)) {
        j.current_journey.current_elemental_state[el] =
          +((j.current_journey.current_elemental_state[el] ?? 0) + d).toFixed(3);
      }
    }
    if (effects?.hexagramShift) {
      j.current_journey.primary_hexagram = { number: effects.hexagramShift.to, shifted: true, from: effects.hexagramShift.from };
    }
    if (effects?.lineChange) {
      j.current_journey.primary_hexagram = effects.lineChange.resultingHexagram;
      j.current_journey.changing_lines = [];
      if (effects.reward) j.current_journey.rewards.push(effects.reward);
    }
    return j;
  }
}
