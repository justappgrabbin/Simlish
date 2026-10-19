// The Oracle's Thread — a Resonance-Driven I Ching Generative Video RPG.
// World module implementing the WorldAdapter world interface
// (enter/handleChoice/getState/save/load). Reuses the existing PlayerSeed +
// BirthDataStore + GameStateStore; no parallel identity/persistence systems.
//
// Flow: birth -> BaZi -> hexagrams -> destiny vector -> generated scene ->
// interact (click/wait/gesture/combine) -> resonance outcome -> world/visual
// consequence -> next situation. There is no game-over: severe distortion
// routes to shadow/underworld content.

import { join } from 'node:path';
import { computeBaZi, GENERATES, CONTROLS } from '../../divination/bazi.js';
import { mapToHexagrams } from '../../divination/hexagramMapper.js';
import { buildDestinyVector } from '../../divination/destinyVector.js';
import { getHexagram } from '../../iching/ichingResolver.js';
import {
  ENTITIES, embed, getEntity, trigramEmbedding, combineEmbeddings,
  GESTURE_TRIGRAMS, TRIGRAM_ELEMENTS, cosineSimilarity,
} from '../../resonance/embeddingSpace.js';
import {
  computeResonance, resolveOutcome, effectParameters, bandForScore,
  hourBranchElement, recordInteraction,
} from '../../resonance/resonanceCore.js';
import { generateSituation, applyRewriteRules } from '../../novelwriter/grammar.js';
import { defaultExpander } from '../../novelwriter/neuralExpander.js';
import { synthesizeVideoPrompt } from '../../novelwriter/videoPromptSynthesizer.js';
import { VideoQueue } from '../../video/queue.js';
import { ProceduralGenerator } from '../../video/generatorClient.js';
import { AssetCache } from '../../video/assetCache.js';
import { JourneyStateManager } from './stateManager.js';
import { shortHash } from '../../util/hash.js';

export const WORLD_NAME = 'oracles-thread';

// The five elements, for the oracle-marking planner.
const FIVE = ['wood', 'fire', 'earth', 'metal', 'water'];

export function create(options = {}) {
  const expander = options.expander ?? defaultExpander; // LLM seam (novelwriter/neuralExpander.js)
  let ctx = null;
  let jm = null;
  let queue = null;
  let cache = null;
  let generator = null;
  let lastEffectParams = null;

  function scenesDir() {
    const dataDir = ctx.adapter.deps.dataDir;
    if (!dataDir) throw new Error('oraclesThread: adapter deps missing dataDir');
    return join(dataDir, 'oracles', 'scenes');
  }

  function journey() { return jm.get(); }
  function elementAnalysis() { return journey().birth_signature.elements; }

  function hexagramOfCurrent() {
    const ph = journey().current_journey.primary_hexagram;
    return { ...getHexagram(ph.number), ...(ph.shifted ? { shifted: true, from: ph.from } : {}) };
  }

  /**
   * Oracle-marking: pick the scene's resonant (intent) entity — the entity the
   * current double-hour "calls" — and its shadow counterpart. For each
   * candidate resonant element we simulate the full action probe (all clicks +
   * all gestures) and keep the choice where both fates stay reachable:
   * clicking the resonant entity can exceed +0.5 (perfect harmony) and the
   * most clashing action can fall below -0.5 (severe distortion).
   */
  function oracleMarkedEntities(now = new Date()) {
    const destiny = journey().current_journey.destiny_vector;
    const { element: hourEl } = hourBranchElement(now);
    const gestures = Object.entries(GESTURE_TRIGRAMS).map(([symbol, tri]) => ({
      label: `gesture:${symbol}`, element: TRIGRAM_ELEMENTS[tri], vec: trigramEmbedding(tri),
    }));
    const clicks = ENTITIES.map((e) => ({ label: `click:${e.name}`, element: e.element, vec: embed(e.name), entity: e }));
    const scoreOf = (cand, intentVector) => computeResonance({
      actionEmbedding: cand.vec, actionElement: cand.element, intentVector, destinyVector: destiny, now,
    }).score;

    let best = null;
    for (const resEl of FIVE) {
      let resonant = null;
      for (const e of ENTITIES.filter((x) => x.element === resEl)) {
        const c = cosineSimilarity(embed(e.name), destiny);
        if (!resonant || c > resonant.c) resonant = { ...e, c };
      }
      const intentVector = embed(resonant.name);
      const maxScore = scoreOf({ element: resonant.element, vec: intentVector }, intentVector); // clicking the resonant entity: cos=1
      let minScore = Infinity, minAction = null;
      for (const cand of [...clicks, ...gestures]) {
        const s = scoreOf(cand, intentVector);
        if (s < minScore) { minScore = s; minAction = cand; }
      }
      const margin = Math.min(maxScore - 0.5, -0.5 - minScore);
      if (!best || margin > best.margin) {
        best = { resonant, intentVector, maxScore, minScore, minAction, margin, hourEl };
      }
    }
    // Shadow entity to inject into the scene: the min action itself if it is a
    // click, else the most destiny-averse entity of the min action's element.
    let shadow;
    if (best.minAction.entity) shadow = best.minAction.entity;
    else {
      shadow = null;
      for (const e of ENTITIES.filter((x) => x.element === best.minAction.element)) {
        const c = cosineSimilarity(embed(e.name), destiny);
        if (!shadow || c < shadow.c) shadow = { ...e, c };
      }
    }
    return { resonant: best.resonant, shadow, intentVector: best.intentVector, minAction: best.minAction, margin: best.margin, hourEl };
  }

  async function prepareScene({ bandHint = null, now = new Date() }) {
    const j = journey();
    const hex = hexagramOfCurrent();
    const sceneSeq = j.current_journey.scenes_visited;
    const seedKey = `${ctx.seed.playerId}:${sceneSeq}:${hex.number}`;

    // 1) Grammar-generated situation + Klein rewrite rules.
    let situation = generateSituation({ seedKey, hexagram: hex, elementAnalysis: elementAnalysis(), bandHint });
    situation = applyRewriteRules(situation, {
      elementAnalysis: elementAnalysis(),
      resultingHexagram: j.current_journey.resulting_hexagram,
      now,
      lastBand: j.session_state.last_band,
    });

    // 2) Oracle-marked entities: the resonant entity the hour calls for, and
    //    the shadow entity the hour resists (guarantees both fates reachable).
    const marked = oracleMarkedEntities(now);
    for (const m of [marked.resonant, marked.shadow]) {
      if (!situation.entities.items.some((e) => e.name === m.name)) {
        situation.entities.items.push({ name: m.name, category: m.category, element: m.element });
      }
    }

    // 3) Prose via the (pluggable) neural expander.
    const hexMeaning = `#${hex.number} ${hex.name} (${hex.upper} over ${hex.lower})`;
    const el = elementAnalysis();
    const destinySummary = `dominant ${el.dominant}, starved ${el.weakest}, thread ${j.birth_signature.pillars.year} ${j.birth_signature.pillars.month} ${j.birth_signature.pillars.day} ${j.birth_signature.pillars.hour}`;
    const prose = await expander.expand({ situation, hexagramMeaning: hexMeaning, destinySummary });

    // 4) Video prompt synthesis + scene DNA.
    const band = bandHint ?? j.session_state.last_band ?? 'resonance';
    const fx = lastEffectParams ?? effectParameters('resonance', 0);
    const synth = synthesizeVideoPrompt({ situation, band, effectParams: fx, elementAnalysis: el });
    const sceneDNA = {
      ...synth,
      band,
      effectParams: fx,
      leadElement: situation.setting.elementScene.element ?? el.dominant,
      situationSeed: seedKey,
    };
    sceneDNA.hash = cache.hashScene(sceneDNA);

    // 5) Queue + generate-or-cache (prompt-hash dedup).
    queue.enqueue({ sceneDNA, priority: 1 });
    let asset = null;
    let job;
    while ((job = queue.dequeue())) {
      const cached = cache.get(job.sceneDNA.hash);
      if (cached) asset = { ...cached, deduped: true };
      else {
        const generated = await generator.generate(job.sceneDNA);
        asset = cache.put(job.sceneDNA.hash, generated, job.sceneDNA);
      }
    }

    // 6) Session state (§7.1).
    const intentEntity = marked.resonant.name;
    jm.beginScene({
      entities: situation.entities.items.map((e) => e.name),
      intentEntity,
      intentVector: marked.intentVector,
      sceneHash: sceneDNA.hash,
      now,
    });

    return { situation, prose, sceneDNA, asset, intentEntity, marked };
  }

  /** Resolve an interaction mode into an action embedding + element. */
  function resolveAction(choice) {
    const session = journey().session_state;
    const mode = choice?.mode;
    if (!mode) throw new Error('oraclesThread: choice.mode required (click|wait|gesture|combine)');
    switch (mode) {
      case 'click': {
        const name = choice.entity;
        if (!session.available_entities.includes(name)) {
          throw new Error(`oraclesThread: entity "${name}" not in scene (have: ${session.available_entities.join(', ')})`);
        }
        return { actionEmbedding: embed(name), actionElement: getEntity(name).element, text: `click:${name}` };
      }
      case 'wait': {
        // Patience path: the stillness entity, earth element, no forcing.
        return { actionEmbedding: embed('stillness'), actionElement: 'earth', text: 'wait:stillness' };
      }
      case 'gesture': {
        const trigram = GESTURE_TRIGRAMS[choice.symbol];
        if (!trigram) throw new Error(`oraclesThread: unknown gesture symbol "${choice.symbol}" (have: ${Object.keys(GESTURE_TRIGRAMS).join(', ')})`);
        return { actionEmbedding: trigramEmbedding(trigram), actionElement: TRIGRAM_ELEMENTS[trigram], text: `gesture:${choice.symbol}->${trigram}` };
      }
      case 'combine': {
        for (const n of [choice.a, choice.b]) {
          if (!session.available_entities.includes(n)) {
            throw new Error(`oraclesThread: entity "${n}" not in scene`);
          }
        }
        const elA = getEntity(choice.a).element, elB = getEntity(choice.b).element;
        // generating pair empowers the generated element; otherwise leads with A.
        const element = elA === elB ? elA : (GENERATES[elA] === elB ? elB : elA);
        return {
          actionEmbedding: combineEmbeddings(embed(choice.a), embed(choice.b)),
          actionElement: element,
          text: `combine:${choice.a}+${choice.b}`,
        };
      }
      default:
        throw new Error(`oraclesThread: unknown mode "${mode}"`);
    }
  }

  return {
    name: WORLD_NAME,

    async enter(context) {
      ctx = context;
      if (!ctx.adapter.deps.birthStore) throw new Error('oraclesThread: adapter deps missing birthStore');
      jm = new JourneyStateManager(ctx.gameState);
      const dir = scenesDir();
      queue = new VideoQueue();
      cache = new AssetCache(dir);
      generator = options.generator ?? new ProceduralGenerator({ scenesDir: dir }); // API seam: Runway/Kling/Wan clients
      if (ctx.gameState.lastScene && jm.exists()) return { ...structuredClone(ctx.gameState.lastScene), resumed: true };

      const birthRecord = ctx.adapter.deps.birthStore.get(ctx.seed.birthDataId);
      const bazi = computeBaZi(birthRecord);

      let resumed = false;
      if (jm.exists()) {
        resumed = true;
        // Resume into an active scene regardless of the phase at save time.
        journey().current_journey.phase = 'SCENE_ACTIVE';
      } else {
        const signature = `${ctx.seed.birthDataId}:${bazi.pillars.year.text}${bazi.pillars.month.text}${bazi.pillars.day.text}${bazi.pillars.hour.text}`;
        const hexagrams = mapToHexagrams(bazi.elements, signature, new Date());
        const destinyVector = buildDestinyVector({
          playerId: ctx.seed.playerId,
          pillars: bazi.pillars,
          elements: bazi.elements,
          hexagrams,
        });
        jm.initialize({
          birthDataId: ctx.seed.birthDataId,
          pillars: bazi.pillars,
          elements: bazi.elements,
          hexagrams,
          destinyVector,
        });
        jm.transition('DIVINING');
        jm.transition('SCENE_ACTIVE');
      }

      const scene = await prepareScene({ now: new Date() });
      return {
        worldName: WORLD_NAME,
        resumed,
        playerId: ctx.seed.playerId,
        bazi: {
          pillars: journey().birth_signature.pillars,
          elements: elementAnalysis(),
        },
        journey: structuredClone(journey()),
        situation: scene.situation,
        prose: scene.prose,
        sceneDNA: scene.sceneDNA,
        asset: scene.asset,
        availableEntities: journey().session_state.available_entities,
        availableGestures: Object.keys(GESTURE_TRIGRAMS),
        modes: ['click', 'wait', 'gesture', 'combine'],
      };
    },

    async handleChoice(choice) {
      if (!ctx) throw new Error('oraclesThread: not entered');
      const j = journey();
      const now = new Date();

      jm.transition('RESOLVING');
      const action = resolveAction(choice);
      const res = computeResonance({
        actionEmbedding: action.actionEmbedding,
        actionElement: action.actionElement,
        intentVector: j.session_state.situation_intent_vector,
        destinyVector: j.current_journey.destiny_vector,
        now,
      });
      const band = bandForScore(res.score);
      const effects = resolveOutcome({
        score: res.score,
        currentHexagram: j.current_journey.primary_hexagram.number,
        changingLines: j.current_journey.changing_lines,
        resultingHexagram: j.current_journey.resulting_hexagram,
        seedKey: `${ctx.seed.playerId}:${j.current_journey.scenes_visited}`,
      });
      const fx = effectParameters(band, res.score);
      lastEffectParams = fx;

      // Elemental state shift: harmony feeds the acted element, distortion starves it.
      const elementDelta = { [action.actionElement]: +(res.score * 0.5).toFixed(3) };

      // History segment of the destiny vector accumulates every interaction.
      const { newDestiny } = recordInteraction(j.current_journey.destiny_vector, action.actionEmbedding);
      jm.recordOutcome({
        score: res.score, band, effects,
        interactionText: action.text,
        newDestinyVector: newDestiny,
        elementDelta,
      });

      // World/visual consequence -> next generated situation + scene asset.
      const scene = await prepareScene({ bandHint: band, now });
      jm.transition('SCENE_ACTIVE');

      return {
        mode: choice.mode,
        interaction: action.text,
        score: res.score,
        components: res.components,
        weights: res.weights,
        band,
        effects,
        effectParams: fx,
        sceneDNA: scene.sceneDNA,
        asset: scene.asset,
        prose: scene.prose,
        situation: scene.situation,
        journey: structuredClone(journey()),
      };
    },

    getState() {
      if (!ctx) throw new Error('oraclesThread: not entered');
      return {
        worldName: WORLD_NAME,
        playerId: ctx.seed.playerId,
        journey: structuredClone(journey()),
        gameState: structuredClone(ctx.gameState),
      };
    },

    save() {
      if (!ctx) throw new Error('oraclesThread: not entered');
      return ctx.adapter.deps.gameStateStore.save(ctx.gameState);
    },

    load() {
      if (!ctx) throw new Error('oraclesThread: not entered');
      ctx.gameState = ctx.adapter.deps.gameStateStore.load(ctx.seed.playerId, WORLD_NAME);
      jm = new JourneyStateManager(ctx.gameState);
      return ctx.gameState;
    },

    // Exposed for probing/tests: current intent + destiny vectors and helpers.
    _internals() {
      return {
        journey: journey(),
        embed, computeResonance, bandForScore, effectParameters,
        hourBranchElement, oracleMarkedEntities,
        shortHash,
      };
    },
  };
}
