// Game One — I Ching Story Adventure (choose-your-own-adventure world module).
// Scenes are GENERATED from world state + resolveIChing output; the I Ching
// resolution manifests as world behavior (river rising, NPC refusing, hidden
// paths appearing). No hexagram explanation is required to play.

import { resolveIChing, getHexagram, TRIGRAMS } from '../iching/ichingResolver.js';
import { seededRng, pickSeeded } from '../util/hash.js';
import { ReactiveLoop } from '../world/reactiveLoop.js';
import { EmbodimentRenderer } from '../render/embodimentRenderer.js';
import { createAddress } from '../state/ontologicalAddress.js';

// PHASE 3: two persistent reactive NPCs live in this world. Their state
// (needs/drives/tensions/relationships/addresses/morphs) persists inside the
// game save under gs.reactiveLoop.
const REACTIVE_NPCS = [
  { id: 'ferryman', purpose: 'guard the crossing', species: 'hybrid', biome: 'forest' },
  { id: 'temple-novice', purpose: 'tend the shrine and help travelers', species: 'hybrid', biome: 'forest' },
];

const SCENE_SETTINGS = ['river ford', 'mountain pass', 'market town', 'reed marsh', 'ancestral shrine', 'border gate'];
const SCENE_NPCS = ['ferryman', 'widow innkeeper', 'traveling merchant', 'old soldier', 'temple novice', 'magistrate'];
const SCENE_OBJECTS = ['bronze mirror', 'sealed letter', 'jade token', 'coiled rope', 'clay jar', 'worn map'];

export function create() {
  let ctx = null;
  let loop = null; // ReactiveLoop instance (state persisted in gs.reactiveLoop)

  // Build/restore the reactive loop; ensure the player's OntologicalAddress.
  function ensureReactiveLoop() {
    const gs = ctx.gameState;
    const dataDir = ctx.adapter.deps.dataDir;
    const deps = {
      worldId: 'iching-story',
      renderer: dataDir ? new EmbodimentRenderer(dataDir) : null,
      playerRef: { entityId: ctx.seed.playerId, identityRef: ctx.seed.visualDNA?.anchors?.anchorsHash ?? ctx.seed.visualIdentityFeatures.fingerprint },
    };
    loop = gs.reactiveLoop ? ReactiveLoop.fromState(gs.reactiveLoop, deps) : new ReactiveLoop(deps);
    for (const npc of REACTIVE_NPCS) loop.registerCharacter(npc);
    // Player's persistent OntologicalAddress lives on the seed.
    if (!ctx.seed.ontologicalAddress) {
      const address = createAddress({
        entityId: ctx.seed.playerId,
        identityRef: deps.playerRef.identityRef,
        worldId: 'iching-story',
        purpose: 'traveler seeking the way',
        lineageMarkers: Object.entries(ctx.seed.ichingState.activeGateStates ?? {}).slice(0, 3)
          .map(([gate, line]) => ({ type: 'gate-mark', symbol: `${gate}.${line}`, gate: +gate, line })),
      });
      ctx.adapter.deps.seedStore.update(ctx.seed.playerId, { ontologicalAddress: address });
      ctx.seed.ontologicalAddress = address;
    }
    return loop;
  }

  function persistReactiveLoop() {
    ctx.gameState.reactiveLoop = loop.getState();
  }

  function worldStateSnapshot() {
    return {
      sceneIndex: ctx.gameState.storyPosition.sceneIndex,
      tick: ctx.gameState.tick,
      environment: ctx.gameState.localWorldState.environment ?? {},
      npcs: ctx.gameState.localRelationships,
      activeHexagram: ctx.gameState.localWorldState.activeHexagram ?? ctx.seed.ichingState.activeHexagram,
      activeLine: ctx.gameState.localWorldState.activeLine ?? ctx.seed.ichingState.activeLine,
    };
  }

  // Generate a scene from world state — deterministic given state.
  function generateScene(extra = {}) {
    const ws = worldStateSnapshot();
    const rng = seededRng(`scene:${ctx.seed.playerId}:${ws.sceneIndex}:${ws.activeHexagram}:${ws.activeLine}`);
    const setting = pickSeeded(SCENE_SETTINGS, rng);
    const npc = pickSeeded(SCENE_NPCS, rng);
    const object = pickSeeded(SCENE_OBJECTS, rng);
    const hex = getHexagram(ws.activeHexagram);
    const envEffects = ctx.gameState.localWorldState.lastEnvironmentalEffects ?? [];

    // The world BEHAVES according to the last I Ching resolution.
    const manifestations = envEffects.map((e) => `${e.change} (${e.type}, strength ${e.magnitude})`);
    const pressure = ctx.gameState.localWorldState.narrativePressure ?? 'quiet-before-change';

    return {
      sceneIndex: ws.sceneIndex,
      setting,
      situation: `At the ${setting}, a ${npc} waits near a ${object}. ${manifestations.length ? 'Around you: ' + manifestations.join('; ') + '.' : 'The air is still.'}`,
      characters: [{ id: npc, disposition: dispositionOf(npc) }],
      objects: [object],
      conflict: `The pressure of the moment is ${pressure}.`,
      choices: buildChoices(ws, rng),
      worldBehavior: manifestations,
      hexagramInPlay: { number: hex.number, name: hex.name, line: ws.activeLine }, // optional flavor, not required to play
      ...extra,
    };
  }

  function dispositionOf(npcId) {
    const rel = ctx.gameState.localRelationships.find((r) => r.id === npcId);
    return rel?.disposition ?? 0;
  }

  function buildChoices(ws, rng) {
    const base = [
      { id: 'approach', label: 'Approach and speak plainly' },
      { id: 'observe', label: 'Watch quietly before acting' },
      { id: 'offer', label: 'Offer something before asking' },
      { id: 'confront', label: 'Confront and press your way through' },
      { id: 'depart', label: 'Leave this place now' },
    ];
    // Actions surfaced by the last resolution become real choices.
    const extra = (ctx.gameState.localWorldState.availableActions ?? []).map((a, i) => ({
      id: `iching-${i}`, label: a,
    }));
    // PHASE 3: NPC behavior options — each reactive character's current
    // disposition/last action surfaces as a real, playable choice.
    const npcChoices = loop ? [...loop.characters.values()].map((c) => ({
      id: `npc-engage-${c.id}`,
      label: `Engage the ${c.id} (${c.lastAction ? c.lastAction.label : 'waits'}; disposition ${c.relationshipMap.player > 0 ? '+' : ''}${c.relationshipMap.player}, form: ${c.form.species})`,
    })) : [];
    return [...base, ...extra, ...npcChoices];
  }

  return {
    name: 'iching-story',

    enter(context) {
      ctx = context;
      const gs = ctx.gameState;
      if (gs.lastScene) { ensureReactiveLoop(); return structuredClone(gs.lastScene); }
      if (!gs.localWorldState.activeHexagram) {
        gs.localWorldState = {
          ...gs.localWorldState,
          activeHexagram: ctx.seed.ichingState.activeHexagram,
          activeLine: ctx.seed.ichingState.activeLine,
          environment: { weather: 'clear', terrain: 'dry road', riverLevel: 1.0 },
        };
      }
      // Entry morph: wanderer embodiment projected from identity.
      const { morph } = ctx.adapter.morphAndRender(ctx.seed, {
        gameId: 'iching-story', sceneId: `scene-${gs.storyPosition.sceneIndex}`, kind: 'wanderer',
      });
      gs.currentMorph = { embodimentId: morph.embodimentId, kind: morph.kind };
      ensureReactiveLoop();
      persistReactiveLoop();
      return generateScene({ entered: true });
    },

    handleChoice(choice) {
      if (!ctx) throw new Error('ichingStory: not entered');
      const gs = ctx.gameState;
      const selected = typeof choice === 'string' ? choice : choice?.id;
      const allowed = buildChoices(worldStateSnapshot(), null).map(option => option.id);
      if (!allowed.includes(selected)) throw new Error('ichingStory: choose an available action');
      const ws = worldStateSnapshot();

      // SHARED resolver — all I Ching logic lives in ichingResolver.js.
      const resolution = resolveIChing({
        playerState: { playerId: ctx.seed.playerId, progression: ctx.seed.persistentProgression },
        worldState: ws,
        relationshipState: { npcs: gs.localRelationships.length ? gs.localRelationships : [{ id: 'ferryman' }] },
        activeHexagram: ws.activeHexagram,
        activeLine: ws.activeLine,
        previousState: { hexagram: ws.activeHexagram, sceneIndex: ws.sceneIndex },
        event: 'player-choice',
        choice: typeof choice === 'string' ? choice : choice?.id,
        temporalState: { tick: gs.tick },
      });

      // --- Resolution becomes WORLD BEHAVIOR (observable state mutation). ---
      const env = { ...(gs.localWorldState.environment ?? {}) };
      for (const eff of resolution.environmentalEffects) {
        if (eff.type === 'weather') env.weather = eff.change;
        if (eff.type === 'terrain') env.terrain = eff.change;
        if (eff.type === 'npc') env.npcEvent = eff.change;
        if (eff.type === 'object') env.objectEvent = eff.change;
      }
      // River rises concretely when water is in play.
      const hex = getHexagram(ws.activeHexagram);
      if (hex.upper === 'kan' || hex.lower === 'kan') {
        env.riverLevel = +((env.riverLevel ?? 1) + 0.3 * (resolution.environmentalEffects[0]?.magnitude ?? 1)).toFixed(3);
      }

      // Relationships shift per resolution.
      const rels = [...gs.localRelationships];
      for (const re of resolution.relationshipEffects) {
        const i = rels.findIndex((r) => r.id === re.npc);
        if (i >= 0) rels[i] = { ...rels[i], disposition: +(rels[i].disposition + re.dispositionShift).toFixed(2) };
        else rels.push({ id: re.npc, disposition: re.dispositionShift, note: re.note });
      }

      // Player morphs as consequence — flowing through the shared morph engine.
      const me = resolution.morphEffects[0];
      const { morph } = ctx.adapter.morphAndRender(ctx.seed, {
        gameId: 'iching-story',
        sceneId: `scene-${gs.storyPosition.sceneIndex}`,
        kind: me.kind,
        environmentalInfluence: { pressure: resolution.narrativePressure, amplification: me.amplification, trigrams: me.trigramSignature },
      });

      // Advance state.
      gs.tick += 1;
      gs.storyPosition.sceneIndex += 1;
      gs.localWorldState = {
        ...gs.localWorldState,
        environment: env,
        activeHexagram: resolution.nextState.hexagram, // state movement
        activeLine: ((ws.activeLine % 6) + 1),
        narrativePressure: resolution.narrativePressure,
        lastEnvironmentalEffects: resolution.environmentalEffects,
        availableActions: resolution.availableActions,
        spawnedEvents: [...(gs.localWorldState.spawnedEvents ?? []), ...resolution.spawnedEvents],
        probabilityAdjustments: resolution.probabilityAdjustments,
      };
      gs.localRelationships = rels;
      gs.ichingNextState = resolution.nextState;
      gs.currentMorph = { embodimentId: morph.embodimentId, kind: morph.kind };

      // PHASE 3: the living world ticks. NPCs perceive the player's action and
      // each other, act, maybe morph, and record history + address updates.
      const choiceId = typeof choice === 'string' ? choice : choice?.id;
      const npcTarget = choiceId?.startsWith('npc-engage-') ? choiceId.slice('npc-engage-'.length) : null;
      const tickResult = loop.tick({
        worldState: { ...worldStateSnapshot(), activeHexagram: resolution.nextState.hexagram, activeLine: gs.localWorldState.activeLine },
        playerAction: { id: choiceId, target: npcTarget },
        tick: gs.tick,
      });
      // NPC world notes become observable environment behavior.
      if (tickResult.worldNotes.length) env.npcActivity = tickResult.worldNotes.join('; ');
      gs.localWorldState.environment = env;
      gs.localRelationships = [...gs.localRelationships.filter((r) => !loop.characters.has(r.id)),
        ...[...loop.characters.values()].map((c) => ({ id: c.id, disposition: c.relationshipMap.player, note: c.lastAction?.label }))];
      persistReactiveLoop();

      return { resolution, scene: generateScene(), morph: { embodimentId: morph.embodimentId, kind: morph.kind }, npcTick: { actions: tickResult.actions, chainEvents: tickResult.chainEvents, morphs: tickResult.morphs } };
    },

    getState() {
      if (!ctx) throw new Error('ichingStory: not entered');
      return { worldName: 'iching-story', ...structuredClone(ctx.gameState) };
    },

    save() {
      if (!ctx) throw new Error('ichingStory: not entered');
      return ctx.adapter.deps.gameStateStore.save(ctx.gameState);
    },

    load() {
      if (!ctx) throw new Error('ichingStory: not entered');
      ctx.gameState = ctx.adapter.deps.gameStateStore.load(ctx.seed.playerId, 'iching-story');
      return ctx.gameState;
    },
  };
}
