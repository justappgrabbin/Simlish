// Game Two — Human Design RPG (tabletop-RPG world module).
// The canonical chart is PROJECTED into mechanics via the interpretation layer
// below: centers -> attributes, channels -> pathways, gates -> abilities
// (game-local names; canonical gate identity preserved), lines modify behavior,
// authority drives decision mechanics, profile drives NPC interaction,
// definition drives connection mechanics, undefined centers amplify/sensitize.
// Exaggeration flows through morphEngine: glowing centers, literal gate
// doorways, channels as routes, tension as creatures.

import { seededRng, shortHash } from '../util/hash.js';

// Interpretation layer: canonical gate -> game-local ability manifestation.
const GATE_ABILITIES = {
  31: { name: 'Voice of the Alpha', school: 'leadership', effect: 'command NPCs once per scene' },
  41: { name: 'Spark of Beginnings', school: 'initiation', effect: 'start a new quest thread' },
  27: { name: 'Warden’s Care', school: 'nourishment', effect: 'restore ally resolve' },
  34: { name: 'Unbending Power', school: 'might', effect: 'overpower one obstacle' },
  20: { name: 'Word of Now', school: 'presence', effect: 'act without delay' },
  28: { name: 'Stakes of Mortality', school: 'risk', effect: 'double reward or double loss' },
  38: { name: 'The Fighter’s Stand', school: 'defiance', effect: 'resist forced choices' },
  50: { name: 'Keeper of Values', school: 'guardianship', effect: 'protect the party’s supplies' },
  7:  { name: 'Banner of Order', school: 'guidance', effect: 'set the party formation' },
  30: { name: 'Fire of Longing', school: 'desire', effect: 'sense the nearest goal' },
};

// Canonical center -> RPG attribute system.
const CENTER_ATTRIBUTES = {
  head: 'inspiration', ajna: 'analysis', throat: 'expression', g: 'direction',
  heart: 'willpower', sacral: 'vitality', solarPlexus: 'emotion', spleen: 'instinct', root: 'drive',
};

const LINE_MODIFIERS = {
  1: { suffix: 'of Foundations', behavior: 'steady, low variance' },
  2: { suffix: 'of the Hermit', behavior: 'stronger when unsolicited' },
  3: { suffix: 'of Trial', behavior: 'may backfire, teaches on failure' },
  4: { suffix: 'of Networks', behavior: 'stronger with allies nearby' },
  5: { suffix: 'of Projection', behavior: 'high expectations from NPCs' },
  6: { suffix: 'of the Roof', behavior: 'dormant early, peaks late' },
};

export function create() {
  let ctx = null;

  // Build mechanics purely from the seed's canonical chart.
  function projectChart() {
    const hd = ctx.seed.humanDesignState;

    const attributes = {};
    for (const [center, info] of Object.entries(hd.centers)) {
      attributes[CENTER_ATTRIBUTES[center]] = {
        center,
        defined: info.defined,
        value: info.defined ? 10 : 4,              // undefined centers = weaker base
        sensitivity: info.defined ? 0.2 : 1.0,     // ...but amplified sensitivity
      };
    }

    const abilities = hd.gates.map((g) => {
      const manifest = GATE_ABILITIES[g.gate] ?? { name: `Gate ${g.gate} Manifestation`, school: 'general', effect: 'unlisted power' };
      const lineMod = LINE_MODIFIERS[g.line];
      return {
        canonicalGate: g.gate, canonicalLine: g.line, center: g.center, // canonical identity preserved
        name: `${manifest.name} ${lineMod.suffix}`,
        school: manifest.school,
        effect: manifest.effect,
        lineBehavior: lineMod.behavior,
        power: 5 + g.line + (hd.centers[g.center]?.defined ? 3 : 0),
      };
    });

    const pathways = hd.channels.map((ch) => ({
      channel: ch.id, name: ch.name,
      route: ch.centers.map((c) => CENTER_ATTRIBUTES[c]),
      traversalCost: ch.centers.every((c) => hd.centers[c]?.defined) ? 1 : 3,
    }));

    return {
      type: hd.type,
      authority: hd.authority, // decision mechanic: sacral = respond, not initiate
      profile: hd.profile,     // NPC interaction modifier
      definition: hd.definition, // connection mechanic
      decisionMechanic: hd.authority === 'Sacral' ? 'respond-to-prompts' : 'initiate',
      npcInteractionStyle: hd.profile === '3/5' ? 'trial-and-error magnetism' : 'steady rapport',
      connectionMechanic: hd.definition === 'Split Definition' ? 'needs-bridging-ally' : 'self-contained',
      attributes,
      abilities,
      pathways,
    };
  }

  // Exaggerated embodiment: chart made physically visible via morphEngine.
  function embody(mechanics, kind = 'gift', extra = {}) {
    const chartAmplifications = {};
    for (const [center, info] of Object.entries(ctx.seed.humanDesignState.centers)) {
      chartAmplifications[center] = info.defined ? 1.0 : 0.25;
    }
    return ctx.adapter.morphAndRender(ctx.seed, {
      gameId: 'hd-rpg',
      sceneId: `encounter-${ctx.gameState.tick}`,
      kind,
      chartAmplifications,
      equipment: mechanics.abilities.slice(0, 3).map((a) => a.name),
      extra,
    });
  }

  return {
    name: 'hd-rpg',

    enter(context) {
      ctx = context;
      const gs = ctx.gameState;
      if (gs.lastScene) return { ...structuredClone(gs.lastScene), mechanics: projectChart() };
      const mechanics = projectChart();
      gs.localWorldState.mechanicsSummary = {
        abilityCount: mechanics.abilities.length,
        pathwayCount: mechanics.pathways.length,
        definedAttributes: Object.values(mechanics.attributes).filter((a) => a.defined).length,
      };
      gs.currentQuest = gs.currentQuest ?? { id: 'q-first-trial', title: 'The First Trial of Self', stage: 0 };
      const { morph } = embody(mechanics, 'gift', { reason: 'awakening' });
      gs.currentMorph = { embodimentId: morph.embodimentId, kind: morph.kind };
      return {
        scene: 'The tabletop is set. Your chart lies upon it as a living map — centers glowing, channels as roads, gates as doorways.',
        mechanics,
        morph: { embodimentId: morph.embodimentId, kind: morph.kind },
      };
    },

    // choice = {type: 'use-ability', abilityGate} | {type: 'traverse', channel} | {type: 'confront', target}
    handleChoice(choice) {
      if (!ctx) throw new Error('hdRpg: not entered');
      const gs = ctx.gameState;
      const mechanics = projectChart();
      let consequence;

      if (choice?.type === 'use-ability') {
        const ability = mechanics.abilities.find((a) => a.canonicalGate === choice.abilityGate);
        if (!ability) throw new Error(`hdRpg: no ability for canonical gate ${choice.abilityGate}`);
        // Center amplified embodiment: the used gate's center flares visibly.
        const centerAmp = {};
        for (const [c, info] of Object.entries(ctx.seed.humanDesignState.centers)) centerAmp[c] = info.defined ? 1 : 0.25;
        centerAmp[ability.center] = 1.6; // EXAGGERATION: used center over-glows
        const { morph } = ctx.adapter.morphAndRender(ctx.seed, {
          gameId: 'hd-rpg', sceneId: `encounter-${gs.tick}`, kind: 'center-amplified',
          chartAmplifications: centerAmp, equipment: [ability.name],
          extra: { usedGate: ability.canonicalGate },
        });
        gs.currentMorph = { embodimentId: morph.embodimentId, kind: morph.kind };
        gs.localWorldState.lastAbilityUsed = { gate: ability.canonicalGate, name: ability.name, power: ability.power };
        gs.currentQuest.stage += 1;
        consequence = { kind: 'ability', ability, embodimentChanged: true, embodimentId: morph.embodimentId };
      } else if (choice?.type === 'traverse') {
        const pathway = mechanics.pathways.find(path => path.channel === choice.channel);
        if (!pathway) throw new Error('hdRpg: choose an available pathway');
        gs.localWorldState.location = pathway.name;
        gs.localWorldState.lastPathway = { channel: pathway.channel, cost: pathway.traversalCost };
        gs.currentQuest.stage += 1;
        consequence = { kind: 'traversal', pathway, location: pathway.name };
      } else if (choice?.type === 'confront') {
        // Tension manifests as a creature; authority determines the mechanic.
        const rng = seededRng(`confront:${ctx.seed.playerId}:${gs.tick}`);
        const passed = mechanics.decisionMechanic === 'respond-to-prompts' ? rng() > 0.3 : rng() > 0.5;
        const { morph } = embody(mechanics, passed ? 'warrior' : 'shadow', { confrontation: choice.target });
        gs.currentMorph = { embodimentId: morph.embodimentId, kind: morph.kind };
        gs.localWorldState.lastConfrontation = { target: choice.target, passed };
        if (passed) gs.currentQuest.stage += 2;
        consequence = { kind: 'confrontation', passed, embodimentChanged: true, embodimentId: morph.embodimentId };
      } else {
        throw new Error(`hdRpg: unknown choice type "${choice?.type}"`);
      }

      gs.tick += 1;
      return { consequence, quest: gs.currentQuest, morph: gs.currentMorph, mechanics,
        scene: gs.localWorldState.location ? `You arrive at ${gs.localWorldState.location}. The living map opens new routes.` : 'Your living map responds. Choose an ability, take a pathway, or face the trial.' };
    },

    getMechanics() {
      if (!ctx) throw new Error('hdRpg: not entered');
      return projectChart();
    },

    getState() {
      if (!ctx) throw new Error('hdRpg: not entered');
      return { worldName: 'hd-rpg', ...structuredClone(ctx.gameState) };
    },

    save() {
      if (!ctx) throw new Error('hdRpg: not entered');
      return ctx.adapter.deps.gameStateStore.save(ctx.gameState);
    },

    load() {
      if (!ctx) throw new Error('hdRpg: not entered');
      ctx.gameState = ctx.adapter.deps.gameStateStore.load(ctx.seed.playerId, 'hd-rpg');
      return ctx.gameState;
    },
  };
}
