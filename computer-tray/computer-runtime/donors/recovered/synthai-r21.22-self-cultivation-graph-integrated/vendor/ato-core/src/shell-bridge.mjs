// shell-bridge.mjs
// ---------------------------------------------------------------------------
// NEW FILE (not part of the original ato-core / ATO-Core-Klein-Browser build).
// Connects a host UI (e.g. a chat/terminal shell) to the real, already-verified
// ato-core engine. Written to close two specific, evidence-based gaps found
// while auditing ato-core and ato-mcp-v0_1 against synthia-shell-v1-2.html:
//
//   1. GATE_DATA (real per-gate name/archetype/keynote/circuit/planet content
//      in gate-data.mjs) is not imported by index.mjs or any other src file
//      (confirmed: `grep -rn "gate-data" src/*.mjs` outside gate-data.mjs
//      itself returns nothing). It is real, tested data sitting unwired.
//
//   2. StateSpaceKernel.describe() -- the one function in ato-core that looks
//      like a text->gate matcher -- turned out on actual testing to have no
//      real relationship to gate meaning: feeding it "conflict conflict
//      conflict conflict" does NOT surface Gate 6 ("Conflict"). Its "cue"
//      vector is built from token character codes fed through a hash, not
//      from any real keyword/meaning association. That function is left
//      untouched here (it's covered by ato-core's own test suite and other
//      code may depend on its exact behavior) -- this file adds a real
//      alternative for routing purposes instead of patching it in place.
//
// Nothing in ato-core/src/*.mjs is modified. This file only imports from it.
// ---------------------------------------------------------------------------

import { bootstrapATO } from './index.mjs';
import { GATE_DATA, gateContent } from './gate-data.mjs';

// ---------------------------------------------------------------------------
// Real keyword index over ato-core's own real gate content (name / keynote /
// archetype / circuit / semanticCenter / planet). Built once at module load.
// ---------------------------------------------------------------------------
// Common words that appear inside gate content purely as grammatical filler
// (nearly every archetype string starts with "The ..."). Left unfiltered,
// these produced false-positive matches unrelated to actual meaning --
// caught during verification below ("organize the army..." was matching on
// "the" instead of returning no match). Filtered at index-build time.
const STOPWORDS = new Set(['the', 'a', 'an', 'of', 'and', 'or', 'to', 'in', 'on', 'is', 'it']);

function buildGateIndex() {
  const index = new Map(); // token -> [{ gate, weight }]
  for (const [gateStr, entry] of Object.entries(GATE_DATA)) {
    const gate = Number(gateStr);
    const fields = [
      [entry.name, 3],
      [entry.keynote, 3],
      [entry.archetype, 2],
      [entry.circuit, 1],
      [entry.semanticCenter, 1],
      [entry.planet, 1],
    ];
    for (const [text, weight] of fields) {
      if (!text) continue;
      for (const token of String(text).toLowerCase().match(/[a-z0-9']+/g) || []) {
        if (STOPWORDS.has(token)) continue;
        if (!index.has(token)) index.set(token, []);
        index.get(token).push({ gate, weight });
      }
    }
  }
  return index;
}

const GATE_INDEX = buildGateIndex();

/**
 * Real (non-hash) text -> gate matching. Every returned candidate is backed
 * by an actual shared word between the input and that gate's real recorded
 * content -- returns [] rather than a distance-ranked-but-meaningless list
 * when nothing actually matches, which is itself more honest than always
 * returning a top-8 list regardless of relevance.
 */
export function matchGates(text, { limit = 3 } = {}) {
  const tokens = String(text).toLowerCase().match(/[a-z0-9']+/g) || [];
  const scores = new Map();
  const matchedTokens = new Map();
  for (const token of tokens) {
    const hits = GATE_INDEX.get(token);
    if (!hits) continue;
    for (const { gate, weight } of hits) {
      scores.set(gate, (scores.get(gate) || 0) + weight);
      if (!matchedTokens.has(gate)) matchedTokens.set(gate, new Set());
      matchedTokens.get(gate).add(token);
    }
  }
  return [...scores.entries()]
    .map(([gate, score]) => ({
      gate,
      score,
      matchedOn: [...matchedTokens.get(gate)],
      ...gateContent(gate),
    }))
    .sort((a, b) => b.score - a.score || a.gate - b.gate)
    .slice(0, limit);
}

// ---------------------------------------------------------------------------
// ShellBridge: a thin, real wrapper around a live bootstrapATO() mesh.
// This is what a host UI should call -- no more hand-typed status strings.
// ---------------------------------------------------------------------------
export class ShellBridge {
  constructor(options = {}) {
    this.system = bootstrapATO(options);
  }

  status() {
    const snap = this.system.mesh.snapshot();
    return {
      status: this.system.status,
      automatonCount: snap.automatons.length,
      edgeCount: snap.edges.length,
      toolIds: snap.automatons.map((a) => a.id),
      personalized: this.system.profile.personalized,
    };
  }

  listAutomatons() {
    return this.system.mesh.snapshot().automatons.map((a) => ({
      id: a.id,
      functionalLevel: a.functionalLevel,
      addressKey: a.addressKey,
      calls: a.calls,
    }));
  }

  /**
   * Real conversational entry point. Grounds free text against real gate
   * content (matchGates, above) and feeds the matches into the mesh's own
   * real conversationAutomaton as `contributions` -- which already existed
   * and already knew how to render contributions, but nothing upstream was
   * ever supplying it real ones (verified: with no contributions it always
   * returned "<input> — context remains unresolved.").
   */
  async ask(text) {
    const matches = matchGates(text, { limit: 3 });
    const contributions = matches.map(
      (m) => `Gate ${m.gate} · ${m.name} (${m.archetype}) — matched on: ${m.matchedOn.join(', ')}`
    );
    const automaton = this.system.mesh.automatons.get('conversation');
    const reply = await automaton.call(text, { contributions });
    return { reply, grounded: matches.length > 0, matches };
  }

  /** Explicit structured tool call -- the real, working path for the 8
   *  Klein-tool automatons, whose input contracts are heterogeneous JSON
   *  shapes (see klein-tools.mjs) and cannot be safely reverse-engineered
   *  from free text without a real per-tool argument extractor, which this
   *  pass does not attempt (see project notes: flagged as still open, not
   *  silently assumed solved). */
  async run(automatonId, input) {
    if (!this.system.mesh.automatons.has(automatonId)) {
      return { ok: false, reason: 'UNKNOWN_AUTOMATON', automatonId };
    }
    const automaton = this.system.mesh.automatons.get(automatonId);
    const output = await automaton.call(input, {});
    return { ok: true, automatonId, output };
  }
}

export function createShellBridge(options = {}) {
  return new ShellBridge(options);
}
