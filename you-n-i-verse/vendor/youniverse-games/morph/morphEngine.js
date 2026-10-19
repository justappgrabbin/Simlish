// MorphEngine — projects a PlayerSeed's canonical identity into game-specific
// embodiments. Projections never mutate the seed: a MorphState is derived from
// (visual identity fingerprint + chart state + game context) and is fully
// reproducible — same recorded state hash -> same embodiment.

import { sha256Hex, deriveFloats, shortHash } from '../util/hash.js';
import { runKleinChain, OPERATOR_CHAIN } from './kleinOperators.js';

// Embodiment kinds as transformation RULES over fingerprint + chart, not costumes.
export const EMBODIMENT_KINDS = {
  normal:      { scale: 1.0, auraStrength: 0.2, gateGlow: 0.2, bodyForm: 'human', narrativeRole: 'traveler' },
  adventurer:  { scale: 1.05, auraStrength: 0.4, gateGlow: 0.4, bodyForm: 'human', narrativeRole: 'seeker' },
  warrior:     { scale: 1.2, auraStrength: 0.5, gateGlow: 0.5, bodyForm: 'broad', narrativeRole: 'protector' },
  scholar:     { scale: 0.95, auraStrength: 0.6, gateGlow: 0.7, bodyForm: 'slim', narrativeRole: 'interpreter' },
  mage:        { scale: 1.0, auraStrength: 0.9, gateGlow: 0.9, bodyForm: 'slim', narrativeRole: 'channel' },
  wanderer:    { scale: 0.98, auraStrength: 0.35, gateGlow: 0.3, bodyForm: 'slim', narrativeRole: 'drifter' },
  royal:       { scale: 1.15, auraStrength: 0.8, gateGlow: 0.6, bodyForm: 'broad', narrativeRole: 'ruler' },
  childlike:   { scale: 0.7, auraStrength: 0.5, gateGlow: 0.4, bodyForm: 'small', narrativeRole: 'innocent' },
  elder:       { scale: 0.92, auraStrength: 0.7, gateGlow: 0.5, bodyForm: 'bent', narrativeRole: 'sage' },
  mythic:      { scale: 1.5, auraStrength: 1.0, gateGlow: 1.0, bodyForm: 'radiant', narrativeRole: 'legend' },
  elemental:   { scale: 1.1, auraStrength: 0.95, gateGlow: 0.6, bodyForm: 'fluid', narrativeRole: 'force-of-nature' },
  archetypal:  { scale: 1.25, auraStrength: 0.85, gateGlow: 0.85, bodyForm: 'radiant', narrativeRole: 'hexagram-made-flesh' },
  'center-amplified': { scale: 1.1, auraStrength: 0.7, gateGlow: 1.0, bodyForm: 'human', narrativeRole: 'amplified-center' },
  shadow:      { scale: 1.05, auraStrength: 0.15, gateGlow: 0.15, bodyForm: 'bent', narrativeRole: 'not-self' },
  gift:        { scale: 1.08, auraStrength: 0.75, gateGlow: 0.8, bodyForm: 'human', narrativeRole: 'gift-bearer' },
  evolved:     { scale: 1.3, auraStrength: 0.9, gateGlow: 0.95, bodyForm: 'radiant', narrativeRole: 'future-self' },
  'alternate-world': { scale: 1.0, auraStrength: 0.5, gateGlow: 0.5, bodyForm: 'shifted', narrativeRole: 'other-self' },
  disguise:    { scale: 0.97, auraStrength: 0.05, gateGlow: 0.05, bodyForm: 'human', narrativeRole: 'hidden' },
  armored:     { scale: 1.18, auraStrength: 0.3, gateGlow: 0.3, bodyForm: 'broad', narrativeRole: 'defended' },
};

export class MorphEngine {
  /**
   * @param {object} deps - { seedStore } to record projection references (optional
   *   for pure derivation; hub wires it so morphs update currentEmbodiment).
   */
  constructor(deps = {}) {
    this.seedStore = deps.seedStore ?? null;
  }

  /**
   * Produce a MorphState.
   * @param {PlayerSeed} seed - canonical identity (read-only here)
   * @param {object} ctx - {gameId, sceneId, kind, activeGateStates?, activeLines?,
   *   chartAmplifications?, environmentalInfluence?, equipment?, reversible?, extra?}
   */
  morph(seed, ctx, { record = true } = {}) {
    const kind = ctx.kind ?? 'normal';
    const rules = EMBODIMENT_KINDS[kind];
    if (!rules) throw new Error(`MorphEngine: unknown embodiment kind "${kind}"`);

    const activeGateStates = ctx.activeGateStates ?? seed.ichingState.activeGateStates;
    const activeLines = ctx.activeLines ?? Object.values(activeGateStates);
    const chartAmplifications = ctx.chartAmplifications ?? this._defaultAmplifications(seed);

    // Reproducibility hash: everything that defines this projection.
    const recordedState = {
      fingerprint: seed.visualIdentityFeatures.fingerprint,
      birthDataId: seed.birthDataId,
      gameId: ctx.gameId, sceneId: ctx.sceneId, kind,
      activeGateStates, activeLines, chartAmplifications,
      environmentalInfluence: ctx.environmentalInfluence ?? null,
      equipment: ctx.equipment ?? [],
      extra: ctx.extra ?? null,
      species: ctx.species ?? ctx.extra?.species ?? 'hybrid',
      biome: ctx.biome ?? ctx.environmentalInfluence?.biome ?? 'forest',
      lineageMarkers: ctx.lineageMarkers ?? null,
      reversible: ctx.reversible ?? true,
    };
    const embodimentId = `emb-${shortHash(recordedState, 12)}`;
    const floats = deriveFloats(sha256Hex(recordedState), 12);

    const palette = seed.visualIdentityFeatures.palette;
    const baseIdentity = {
      playerId: seed.playerId,
      name: seed.name,
      fingerprint: seed.visualIdentityFeatures.fingerprint,
      dominantPalette: palette.map((p) => p.hex),
      luminance: seed.visualIdentityFeatures.luminance,
      hdType: seed.humanDesignState.type,
      profile: seed.humanDesignState.profile,
    };

    const visualState = {
      scale: +(rules.scale * (0.9 + floats[0] * 0.2)).toFixed(3),
      auraColor: palette[Math.floor(floats[1] * palette.length)].hex,
      auraStrength: +Math.min(1, rules.auraStrength * (0.8 + floats[2] * 0.4)).toFixed(3),
      bodyColor: palette[Math.floor(floats[3] * palette.length)].hex,
      accentColor: palette[Math.floor(floats[4] * palette.length)].hex,
      bodyForm: rules.bodyForm,
      proportions: {
        headRatio: +(0.18 + floats[5] * 0.1).toFixed(3),
        limbRatio: +(0.4 + floats[6] * 0.2).toFixed(3),
        torsoRatio: +(0.3 + floats[7] * 0.15).toFixed(3),
      },
      // Defined centers glow; glow strength from chart amplification + rule.
      glowingCenters: Object.entries(chartAmplifications)
        .filter(([, v]) => v > 0.5)
        .map(([center, v], i) => ({
          center,
          intensity: +Math.min(1, v * rules.gateGlow).toFixed(3),
          color: palette[(i + Math.floor(floats[8] * palette.length)) % palette.length].hex,
        })),
      gateMarks: Object.entries(activeGateStates).map(([gate, line], i) => ({
        gate: +gate, line,
        angle: +(floats[9] * 360 + i * 36) % 360,
      })),
    };

    const morphState = {
      embodimentId,
      inputContext: structuredClone(ctx),
      playerId: seed.playerId,
      gameId: ctx.gameId,
      sceneId: ctx.sceneId ?? null,
      kind,
      baseIdentity,
      activeGateStates,
      activeLines,
      chartAmplifications,
      environmentalInfluence: ctx.environmentalInfluence ?? {},
      narrativeRole: rules.narrativeRole,
      equipment: ctx.equipment ?? [],
      transformationRules: { kind, ...rules },
      visualState,
      embodimentSpec: null, // filled below when seed carries visualDNA anchors
      resultingAsset: null, // filled by renderer wiring
      reversible: ctx.reversible ?? true,
      stateHash: sha256Hex(recordedState),
    };

    // PHASE 2: when the seed carries visualDNA anchors, run the recorded Klein
    // operator chain. Legacy (visualState-only) path above remains for seeds
    // without DNA. Deterministic: spec depends only on seed anchors + ctx.
    if (seed.visualDNA?.anchors) {
      const species = ctx.species ?? ctx.extra?.species ?? 'hybrid';
      const biome = ctx.biome ?? ctx.environmentalInfluence?.biome ?? 'forest';
      const lineageMarkers = ctx.lineageMarkers ?? Object.entries(activeGateStates).slice(0, 3)
        .map(([gate, line]) => ({ type: 'gate-mark', symbol: `${gate}.${line}`, gate: +gate, line }));
      const embodimentSpec = runKleinChain(seed.visualDNA, { species, biome, lineageMarkers });
      morphState.embodimentSpec = embodimentSpec;
      morphState.transformationRules.operatorChain = [...OPERATOR_CHAIN];
      morphState.transformationRules.species = species;
      morphState.transformationRules.biome = biome;
      morphState.transformationRules.specHash = embodimentSpec.specHash;
    }

    if (record && this.seedStore) this.seedStore.recordTransformation(seed.playerId, morphState);
    return morphState;
  }

  // Defined centers amplify; undefined centers stay sensitive (low base).
  _defaultAmplifications(seed) {
    const out = {};
    for (const [center, info] of Object.entries(seed.humanDesignState.centers)) {
      out[center] = info.defined ? 1.0 : 0.25;
    }
    return out;
  }

  // Re-derive an embodiment from a recorded MorphState's inputs (reproducibility check).
  reproduce(seed, recordedMorphState) {
    const re = this.morph(seed, recordedMorphState.inputContext ?? {
      gameId: recordedMorphState.gameId,
      sceneId: recordedMorphState.sceneId,
      kind: recordedMorphState.kind,
      activeGateStates: recordedMorphState.activeGateStates,
      activeLines: recordedMorphState.activeLines,
      chartAmplifications: recordedMorphState.chartAmplifications,
      environmentalInfluence: recordedMorphState.environmentalInfluence,
      equipment: recordedMorphState.equipment,
      reversible: recordedMorphState.reversible,
      species: recordedMorphState.transformationRules?.species,
      biome: recordedMorphState.transformationRules?.biome,
    }, { record: false });
    return re.stateHash === recordedMorphState.stateHash ? re : null;
  }
}
