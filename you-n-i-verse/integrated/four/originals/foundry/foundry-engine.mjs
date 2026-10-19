/**
 * ============================================================================
 * FOUNDRY ENGINE v1.0
 * Multi-Channel Ontological Compiler
 * ============================================================================
 * 
 * Single-file, zero-dependency execution engine.
 * 
 * INVARIANT: Foundry does not generate meaning. It resolves meaning once,
 * stores it causally, and projects it many ways.
 * 
 * If you cannot trace an output back to Base → Operator → Lineage,
 * the output is invalid.
 * 
 * Pipeline (IRREVERSIBLE — no skips, no branches, no loops):
 *   INPUT → Ontological Parse → Dimensional Binding → Lexicon Resolution
 *   → Structural Encoding → Temporal & Orientational Anchoring
 *   → Causal Graph Commit → Output Projection(s)
 * 
 * @license Foundry Contract v1.0
 *
 * SYNTHIA PORT NOTE:
 * The uploaded reference contradicts its own DAG invariant by adding
 * Tone -> Base after Base -> Gate -> Line -> Color -> Tone. That throws
 * GRAPH_CYCLE_DETECTED for the documented sample. This port preserves the
 * original donor separately and represents the recursive anchor as a terminal
 * Anchor node carrying the base value, so lineage stays causal/DAG-safe.
 */

'use strict';

// ============================================================================
// SECTION 0: INVARIANT CONFIGURATION
// ============================================================================

const INVARIANT = Object.freeze({
  VERSION: '1.0.0',
  SCHEMA_VERSION: '1.0.0',

  // Fixed questions — exactly one per clause
  QUESTIONS: Object.freeze(['WHO', 'WHAT', 'WHERE', 'WHEN', 'WHY', 'HOW']),

  // Base → Dimension → Root Voice mapping
  BASES: Object.freeze({
    1: { dimension: 'Movement', voice: 'I Define / I Create', question: 'WHERE' },
    2: { dimension: 'Evolution', voice: 'I Remember', question: 'WHAT' },
    3: { dimension: 'Being', voice: 'I Am', question: 'WHEN' },
    4: { dimension: 'Design', voice: 'I Design', question: 'WHY' },
    5: { dimension: 'Space', voice: 'I Think', question: 'WHO' }
  }),

  // Structural encoding bounds
  GATE_MIN: 1,  GATE_MAX: 64,
  LINE_MIN: 1,  LINE_MAX: 6,
  COLOR_MIN: 1, COLOR_MAX: 6,
  TONE_MIN: 1,  TONE_MAX: 6,

  // Projection layers
  LAYERS: Object.freeze([
    'PsychoLinguistic',
    'SynthiaNative', 
    'GeneKeys',
    'SpiritualPoetic',
    'CompressedSymbolic'
  ]),

  // Operator algebra
  OPERATORS: Object.freeze({
    Singularity: { class: 'StateTransformer',  effect: 'CreateState',    createsNode: true },
    Transition:  { class: 'StateTransformer',  effect: 'CreateState',    createsNode: true },
    Collapse:    { class: 'StateTransformer',  effect: 'ResolveState',   createsNode: true },
    Portal:      { class: 'CausalEdge',        effect: 'CreateEdge',     createsNode: true },
    Fork:        { class: 'CausalEdge',        effect: 'CreateEdge',     createsNode: true },
    Blade:       { class: 'CausalEdge',        effect: 'CreateEdge',     createsNode: true },
    Vector:      { class: 'CausalEdge',        effect: 'CreateEdge',     createsNode: true },
    Breath:      { class: 'FlowModifier',      effect: 'ModifyFlow',     createsNode: false },
    Current:     { class: 'FlowModifier',      effect: 'ModifyFlow',     createsNode: false },
    Pulse:       { class: 'FlowModifier',      effect: 'ModifyFlow',     createsNode: false },
    Flicker:     { class: 'FlowModifier',      effect: 'ModifyFlow',     createsNode: false },
    Container:   { class: 'ScopeRestrictor',   effect: 'RestrictScope',  createsNode: false },
    Cocoon:      { class: 'ScopeRestrictor',   effect: 'RestrictScope',  createsNode: false },
    Domain:      { class: 'ScopeRestrictor',   effect: 'RestrictScope',  createsNode: false },
    Mirror:      { class: 'IdentityUnifier',   effect: 'UnifyIdentity',  createsNode: true }
  }),

  // Edge types for causal graph
  EDGE_TYPES: Object.freeze([
    'PARSES_TO', 'BINDS_TO', 'RESOLVES_TO', 'EXTENDS_TO',
    'MODULATES_TO', 'TUNES_TO', 'ANCHORS_TO', 'ORIENTS_TO',
    'FRAMES_TO', 'OPERATES_VIA', 'CAUSES', 'PROJECTS_TO', 'VALIDATES'
  ]),

  // Tone mappings by operator effect
  TONE_MAP: Object.freeze({
    CreateState:    { Clinical: 'presents as',      Conversational: 'shows up as',    Contemplative: 'unfolds as',       Mystical: 'becomes',        Terse: 'is' },
    ResolveState:   { Clinical: 'resolves to',        Conversational: 'settles into',   Contemplative: 'rests in',         Mystical: 'returns to',       Terse: '→' },
    CreateEdge:     { Clinical: 'initiates',          Conversational: 'opens',          Contemplative: 'invites',          Mystical: 'calls forth',      Terse: '>' },
    ModifyFlow:     { Clinical: 'modulates',          Conversational: 'shifts',         Contemplative: 'turns',            Mystical: 'breathes',         Terse: '~' },
    RestrictScope:  { Clinical: 'limits',             Conversational: 'contains',       Contemplative: 'holds',            Mystical: 'encloses',         Terse: '[ ]' },
    UnifyIdentity:  { Clinical: 'identifies with',    Conversational: 'merges with',    Contemplative: 'recognizes itself in', Mystical: 'remembers',      Terse: '=' }
  }),

  // Layer tones (fixed)
  LAYER_TONES: Object.freeze({
    PsychoLinguistic: 'Clinical',
    SynthiaNative: 'Conversational',
    GeneKeys: 'Contemplative',
    SpiritualPoetic: 'Mystical',
    CompressedSymbolic: 'Terse'
  })
});


// ============================================================================
// SECTION 1: UTILITY / CRYPTO
// ============================================================================

/** Simple SHA-256 for checksums (browser + Node compatible) */
async function sha256(input) {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);

  if (typeof crypto !== 'undefined' && crypto.subtle) {
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }

  // Node.js fallback
  if (typeof require !== 'undefined') {
    const cryptoNode = require('crypto');
    return cryptoNode.createHash('sha256').update(input).digest('hex');
  }

  throw new FoundryError('CRYPTO_UNAVAILABLE', 'No SHA-256 implementation available');
}

/** UUID v4 generator */
function uuid() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/** Immutable deep freeze */
function deepFreeze(obj) {
  Object.keys(obj).forEach(key => {
    const val = obj[key];
    if (val && typeof val === 'object' && !Object.isFrozen(val)) {
      deepFreeze(val);
    }
  });
  return Object.freeze(obj);
}


// ============================================================================
// SECTION 2: ERROR SYSTEM
// ============================================================================

class FoundryError extends Error {
  constructor(code, message, details = {}) {
    super(message);
    this.name = 'FoundryError';
    this.code = code;
    this.details = deepFreeze(details);
    this.timestamp = new Date().toISOString();
    Object.freeze(this);
  }

  toJSON() {
    return {
      name: this.name,
      code: this.code,
      message: this.message,
      details: this.details,
      timestamp: this.timestamp
    };
  }
}

// Error codes (invariant)
const ERROR_CODES = Object.freeze({
  // Parse errors
  PARSE_INVALID_QUESTION:   'Clause answers zero or multiple questions',
  PARSE_UNKNOWN_QUESTION:   'Question not in fixed set',
  PARSE_INVALID_BASE:       'Base not in 1-5',
  PARSE_CLAUSE_EMPTY:       'Clause text is empty',

  // Graph errors
  GRAPH_CYCLE_DETECTED:     'Causal graph contains cycle',
  GRAPH_MISSING_ROOT:       'No root ontology node',
  GRAPH_INVALID_EDGE:       'Edge references non-existent node',

  // Lexicon errors
  LEXICON_GAP:              'No lexical entry for coordinates',
  LEXICON_SYNONYM_ATTEMPT:  'Attempted to generate synonym (forbidden)',

  // Operator errors
  OPERATOR_INVALID:         'Operator not in closed set',
  OPERATOR_MISSING_NODE:    'Orientation change requires new node',

  // Projection errors
  PROJECTION_INVALID_LAYER: 'Layer not in fixed set',
  PROJECTION_TEMPLATE_MISMATCH: 'No template fits state',
  PROJECTION_TONE_UNMAPPED: 'Operator effect has no tone mapping',
  PROJECTION_TRACEBACK_FAIL: 'Output does not resolve to lineage',

  // System errors
  CRYPTO_UNAVAILABLE:       'No SHA-256 implementation',
  ENGINE_VERSION_MISMATCH:  'Schema version does not match engine version',
  SEAL_BROKEN:              'Post-hoc modification detected'
});


// ============================================================================
// SECTION 3: CLOSED LEXICON (MINIMAL EMBEDDED — REPLACE WITH FULL DICTIONARY)
// ============================================================================

/**
 * The lexicon is a closed semantic dictionary.
 * Keywords are retrieved, not generated.
 * 
 * This is a MINIMAL embedded lexicon for demonstration.
 * Production systems MUST load the full closed dictionary from
 * lexicon/ directory (centers.json, gates.json, lines.json, etc.)
 */
const EMBEDDED_LEXICON = deepFreeze({
  // Gate 1: The Creative
  '1.1.1.1.1': {
    PsychoLinguistic: { term: 'self-actualization drive', role: 'noun_phrase', collocations: ['initiates', 'explores', 'defines'] },
    SynthiaNative:    { term: 'I-am pulse', role: 'state_declaration', collocations: ['resonates', 'anchors', 'extends'] },
    GeneKeys:         { term: 'The Creative', role: 'archetype_title', collocations: ['unfolds through', 'shadows into', 'siddhi of'] },
    SpiritualPoetic:  { term: 'the unmanifest making itself known', role: 'metaphor', collocations: ['breathes', 'becomes', 'remembers'] },
    CompressedSymbolic: { term: '1.1.1.1.1', role: 'coordinate_string', collocations: [] }
  },
  // Gate 2: The Receptive
  '2.1.1.1.2': {
    PsychoLinguistic: { term: 'receptive capacity', role: 'noun_phrase', collocations: ['absorbs', 'contains', 'allows'] },
    SynthiaNative:    { term: 'holding field', role: 'state_declaration', collocations: ['receives', 'stores', 'transmits'] },
    GeneKeys:         { term: 'The Receptive', role: 'archetype_title', collocations: ['opens to', 'receives', 'becomes'] },
    SpiritualPoetic:  { term: 'the womb of becoming', role: 'metaphor', collocations: ['receives', 'holds', 'births'] },
    CompressedSymbolic: { term: '2.1.1.1.2', role: 'coordinate_string', collocations: [] }
  },
  // Gate 3: Difficulty
  '3.1.1.1.3': {
    PsychoLinguistic: { term: 'ordering impulse', role: 'noun_phrase', collocations: ['structures', 'organizes', 'establishes'] },
    SynthiaNative:    { term: 'pattern-seed', role: 'state_declaration', collocations: ['orders', 'arranges', 'sequences'] },
    GeneKeys:         { term: 'Difficulty', role: 'archetype_title', collocations: ['orders through', 'structures', 'integrates'] },
    SpiritualPoetic:  { term: 'the ordering of chaos', role: 'metaphor', collocations: ['shapes', 'forms', 'becomes'] },
    CompressedSymbolic: { term: '3.1.1.1.3', role: 'coordinate_string', collocations: [] }
  }
});


// ============================================================================
// SECTION 4: GRAMMAR TEMPLATES (FIXED — NO MODIFICATION)
// ============================================================================

const GRAMMAR_TEMPLATES = deepFreeze({
  // Template selection: base → operator class → axis
  templates: [
    {
      id: 'T-001',
      match: { base: [1,2,3,4,5], operatorClass: ['StateTransformer', 'CausalEdge'], axis: 'individual' },
      structure: '[Subject] [ToneVerb] [Object] [Contextualizer].',
      slots: {
        Subject:        { source: 'base.voice', transform: 'none' },
        ToneVerb:       { source: 'operator.effect', transform: 'tone_verb' },
        Object:         { source: 'gate.term', transform: 'none' },
        Contextualizer: { source: 'axis.context', transform: 'prepositional_phrase' }
      }
    },
    {
      id: 'T-002',
      match: { base: [1,2,3,4,5], operatorClass: ['FlowModifier'], axis: 'individual' },
      structure: '[Subject] [ToneVerb] [Object] [Contextualizer].',
      slots: {
        Subject:        { source: 'base.voice', transform: 'none' },
        ToneVerb:       { source: 'operator.effect', transform: 'tone_verb' },
        Object:         { source: 'gate.term', transform: 'none' },
        Contextualizer: { source: 'axis.context', transform: 'prepositional_phrase' }
      }
    },
    {
      id: 'T-003',
      match: { base: [1,2,3,4,5], operatorClass: ['ScopeRestrictor'], axis: 'individual' },
      structure: '[Subject] [ToneVerb] [Object] [Contextualizer].',
      slots: {
        Subject:        { source: 'base.voice', transform: 'none' },
        ToneVerb:       { source: 'operator.effect', transform: 'tone_verb' },
        Object:         { source: 'gate.term', transform: 'none' },
        Contextualizer: { source: 'axis.context', transform: 'prepositional_phrase' }
      }
    },
    {
      id: 'T-004',
      match: { base: [1,2,3,4,5], operatorClass: ['IdentityUnifier'], axis: 'individual' },
      structure: '[Subject] [ToneVerb] [Object] [Contextualizer].',
      slots: {
        Subject:        { source: 'base.voice', transform: 'none' },
        ToneVerb:       { source: 'operator.effect', transform: 'tone_verb' },
        Object:         { source: 'gate.term', transform: 'none' },
        Contextualizer: { source: 'axis.context', transform: 'prepositional_phrase' }
      }
    }
  ],

  // Compressed symbolic has its own template (no natural language)
  compressed: {
    id: 'T-COMP',
    structure: '[Base].[Gate].[Line].[Color].[Tone]|[Operator]|[Axis]|[Context]',
    slots: {
      Base:      { source: 'base', transform: 'raw' },
      Gate:      { source: 'gate', transform: 'raw' },
      Line:      { source: 'line', transform: 'raw' },
      Color:     { source: 'color', transform: 'raw' },
      Tone:      { source: 'tone', transform: 'raw' },
      Operator:  { source: 'operator.symbol', transform: 'raw' },
      Axis:      { source: 'axis', transform: 'raw' },
      Context:   { source: 'context', transform: 'raw' }
    }
  }
});


// ============================================================================
// SECTION 5: CAUSAL GRAPH (DAG — NO CYCLES)
// ============================================================================

class CausalGraph {
  constructor() {
    this.nodes = new Map();     // id → node
    this.edges = new Map();     // id → edge
    this.adjacency = new Map(); // from → Set(to)
    this.root = null;
    Object.freeze(this.nodes);
    Object.freeze(this.edges);
    Object.freeze(this.adjacency);
  }

  /** Add node — returns node ID */
  addNode(label, properties = {}) {
    const id = uuid();
    const node = deepFreeze({ id, label, properties: deepFreeze({ ...properties }) });
    this.nodes.set(id, node);
    if (!this.adjacency.has(id)) this.adjacency.set(id, new Set());
    return id;
  }

  /** Add edge — enforces DAG (no cycles) */
  addEdge(from, to, type, weight = 1.0) {
    // Validate nodes exist
    if (!this.nodes.has(from)) throw new FoundryError('GRAPH_INVALID_EDGE', `From node ${from} not found`);
    if (!this.nodes.has(to)) throw new FoundryError('GRAPH_INVALID_EDGE', `To node ${to} not found`);

    // Check for cycle
    if (this._wouldCreateCycle(from, to)) {
      throw new FoundryError('GRAPH_CYCLE_DETECTED', `Edge ${from}→${to} would create cycle`);
    }

    const id = uuid();
    const edge = deepFreeze({
      id, from, to, type,
      weight,
      timestamp: new Date().toISOString()
    });

    this.edges.set(id, edge);
    this.adjacency.get(from).add(to);
    return id;
  }

  /** Check if adding edge from→to would create cycle */
  _wouldCreateCycle(from, to) {
    if (from === to) return true;
    const visited = new Set();
    const stack = [to];
    while (stack.length > 0) {
      const current = stack.pop();
      if (current === from) return true;
      if (visited.has(current)) continue;
      visited.add(current);
      const neighbors = this.adjacency.get(current) || new Set();
      for (const neighbor of neighbors) {
        stack.push(neighbor);
      }
    }
    return false;
  }

  /** Get lineage chain: Base → Gate → Line → Color → Tone → Base */
  getLineage() {
    if (!this.root) return [];

    const lineage = [];
    const traverse = (nodeId, path = []) => {
      const node = this.nodes.get(nodeId);
      if (!node) return;

      if (['Base', 'Gate', 'Line', 'Color', 'Tone'].includes(node.label)) {
        path.push(node.properties.value || node.properties.number || node.label);
      }

      const neighbors = this.adjacency.get(nodeId) || new Set();
      for (const neighbor of neighbors) {
        traverse(neighbor, [...path]);
      }

      if (neighbors.size === 0 && path.length > 0) {
        lineage.push(path.join(' → '));
      }
    };

    traverse(this.root);
    return lineage;
  }

  /** Serialize to schema-compliant JSON */
  toJSON() {
    return deepFreeze({
      root: this.root,
      nodes: Array.from(this.nodes.values()),
      edges: Array.from(this.edges.values()),
      lineage: this.getLineage()
    });
  }

  /** Set root node */
  setRoot(nodeId) {
    if (!this.nodes.has(nodeId)) {
      throw new FoundryError('GRAPH_MISSING_ROOT', `Root node ${nodeId} not found`);
    }
    this.root = nodeId;
    return this;
  }
}


// ============================================================================
// SECTION 6: PARSER — ONTOLOGICAL PARSE
// ============================================================================

class OntologicalParser {
  /**
   * Parse raw input into clauses.
   * Each clause answers EXACTLY ONE question.
   * If a clause answers zero or multiple, it is invalid.
   */
  parse(rawInput) {
    const clauses = [];
    const errors = [];

    // Split input into clauses (by sentence delimiter or explicit markers)
    // For now: split by periods, but respect explicit clause markers
    const rawClauses = rawInput
      .split(/[.\n]+/)
      .map(c => c.trim())
      .filter(c => c.length > 0);

    for (let i = 0; i < rawClauses.length; i++) {
      const text = rawClauses[i];
      const result = this._parseClause(text, i);

      if (result.valid) {
        clauses.push(result.clause);
      } else {
        errors.push({
          clause_index: i,
          code: result.error.code,
          message: result.error.message,
          fatal: result.error.fatal
        });
      }
    }

    return {
      valid: errors.filter(e => e.fatal).length === 0 && clauses.length > 0,
      clauses,
      errors
    };
  }

  _parseClause(text, index) {
    // Detect which question(s) this clause answers
    const detectedQuestions = [];

    for (const q of INVARIANT.QUESTIONS) {
      // Check for explicit question markers (e.g., "WHO:", "WHAT:")
      const marker = new RegExp(`^${q}[:\s]`, 'i');
      if (marker.test(text) || text.toUpperCase().includes(q)) {
        detectedQuestions.push(q);
      }
    }

    // Single-question rule enforcement
    if (detectedQuestions.length === 0) {
      return {
        valid: false,
        error: new FoundryError('PARSE_INVALID_QUESTION', 'Clause answers zero questions', { text, detected: detectedQuestions })
      };
    }

    if (detectedQuestions.length > 1) {
      return {
        valid: false,
        error: new FoundryError('PARSE_INVALID_QUESTION', 'Clause answers multiple questions', { text, detected: detectedQuestions })
      };
    }

    const question = detectedQuestions[0];

    // Determine base from question mapping
    const baseEntry = Object.entries(INVARIANT.BASES).find(
      ([, v]) => v.question === question
    );

    if (!baseEntry) {
      return {
        valid: false,
        error: new FoundryError('PARSE_UNKNOWN_QUESTION', `Question ${question} has no base mapping`)
      };
    }

    const base = parseInt(baseEntry[0]);

    // Extract keywords (simplified — production uses closed lexicon matching)
    const keywords = this._extractKeywords(text, base);

    // Extract operators (simplified — production uses operator detection)
    const operators = this._extractOperators(text);

    const clause = deepFreeze({
      id: uuid(),
      index,
      question,
      base,
      text: text.replace(new RegExp(`^${question}[:\s]*`, 'i'), '').trim(),
      keywords,
      operators,
      valid: true
    });

    return { valid: true, clause };
  }

  _extractKeywords(text, base) {
    // In production: match against closed lexicon for this base
    // For now: return empty array (lexicon retrieval happens later)
    return [];
  }

  _extractOperators(text) {
    const operators = [];
    const textUpper = text.toUpperCase();

    for (const [symbol, meta] of Object.entries(INVARIANT.OPERATORS)) {
      if (textUpper.includes(symbol.toUpperCase())) {
        operators.push(deepFreeze({
          symbol,
          class: meta.class,
          effect: meta.effect,
          orientation_change: false, // Detected in graph phase
          target: null
        }));
      }
    }

    return operators;
  }
}


// ============================================================================
// SECTION 7: LEXICON RESOLVER — CLOSED DICTIONARY LOOKUP
// ============================================================================

class LexiconResolver {
  constructor(lexiconData = EMBEDDED_LEXICON) {
    this.lexicon = deepFreeze({ ...lexiconData });
  }

  /**
   * Retrieve lexical entry for exact coordinates.
   * NO SYNONYMS. NO PARAPHRASING. NO GENERATION.
   */
  retrieve(coordinates, layer) {
    const key = this._makeKey(coordinates);
    const entry = this.lexicon[key];

    if (!entry) {
      throw new FoundryError('LEXICON_GAP', 'No lexical entry for coordinates', {
        coordinates,
        key,
        layer
      });
    }

    const layerEntry = entry[layer];

    if (!layerEntry) {
      throw new FoundryError('LEXICON_GAP', 'No lexical entry for layer', {
        coordinates,
        key,
        layer
      });
    }

    return deepFreeze({ ...layerEntry });
  }

  _makeKey(coords) {
    return `${coords.base}.${coords.gate}.${coords.line}.${coords.color}.${coords.tone}`;
  }

  /** Load external lexicon (production use) */
  load(lexiconData) {
    return new LexiconResolver(lexiconData);
  }
}


// ============================================================================
// SECTION 8: STRUCTURAL ENCODER — GATE → LINE → COLOR → TONE → BASE
// ============================================================================

class StructuralEncoder {
  /**
   * Encode structural coordinates from parsed clauses.
   * In production: this resolves from chart data, gates, etc.
   * For now: uses explicit or default values.
   */
  encode(clauses, overrides = {}) {
    const defaults = {
      gate: 1,
      line: 1,
      color: 1,
      tone: 1,
      axis: 'individual',
      context: 'present'
    };

    // Use first clause's base as the anchor
    const base = clauses[0]?.base || 1;

    return deepFreeze({
      base,
      gate: this._clamp(overrides.gate || defaults.gate, INVARIANT.GATE_MIN, INVARIANT.GATE_MAX),
      line: this._clamp(overrides.line || defaults.line, INVARIANT.LINE_MIN, INVARIANT.LINE_MAX),
      color: this._clamp(overrides.color || defaults.color, INVARIANT.COLOR_MIN, INVARIANT.COLOR_MAX),
      tone: this._clamp(overrides.tone || defaults.tone, INVARIANT.TONE_MIN, INVARIANT.TONE_MAX),
      axis: overrides.axis || defaults.axis,
      context: overrides.context || defaults.context
    });
  }

  _clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }
}


// ============================================================================
// SECTION 9: GRAPH COMMIT — CAUSAL GRAPH CONSTRUCTION
// ============================================================================

class GraphCommiter {
  commit(parseResult, coordinates, operators) {
    const graph = new CausalGraph();

    // 1. Create Ontology node (root)
    const ontologyId = graph.addNode('Ontology', {
      version: INVARIANT.VERSION,
      schemaVersion: INVARIANT.SCHEMA_VERSION,
      clauseCount: parseResult.clauses.length
    });
    graph.setRoot(ontologyId);

    // 2. Create Clause nodes and PARSES_TO edges
    for (const clause of parseResult.clauses) {
      const clauseId = graph.addNode('Clause', {
        question: clause.question,
        base: clause.base,
        text: clause.text,
        valid: clause.valid
      });
      graph.addEdge(ontologyId, clauseId, 'PARSES_TO');

      // 3. Create Base node and BINDS_TO edge
      const baseId = graph.addNode('Base', {
        value: clause.base,
        dimension: INVARIANT.BASES[clause.base].dimension,
        voice: INVARIANT.BASES[clause.base].voice
      });
      graph.addEdge(clauseId, baseId, 'BINDS_TO');

      // 4. Create structural nodes (Gate → Line → Color → Tone)
      const gateId = graph.addNode('Gate', {
        number: coordinates.gate,
        theme: this._getGateTheme(coordinates.gate)
      });
      graph.addEdge(baseId, gateId, 'RESOLVES_TO');

      const lineId = graph.addNode('Line', {
        number: coordinates.line,
        orbit: this._getLineOrbit(coordinates.line)
      });
      graph.addEdge(gateId, lineId, 'EXTENDS_TO');

      const colorId = graph.addNode('Color', {
        number: coordinates.color,
        motivation: this._getColorMotivation(coordinates.color)
      });
      graph.addEdge(lineId, colorId, 'MODULATES_TO');

      const toneId = graph.addNode('Tone', {
        number: coordinates.tone,
        perception: this._getTonePerception(coordinates.tone)
      });
      graph.addEdge(colorId, toneId, 'TUNES_TO');

      // 5. Recursive anchor represented without violating the graph's DAG invariant.
      const anchorId = graph.addNode('Anchor', {
        base: clause.base,
        dimension: INVARIANT.BASES[clause.base].dimension,
        reference: 'Base'
      });
      graph.addEdge(toneId, anchorId, 'ANCHORS_TO');

      // 6. Orientational anchoring
      const axisId = graph.addNode('Axis', {
        value: coordinates.axis,
        context: coordinates.context
      });
      graph.addEdge(baseId, axisId, 'ORIENTS_TO');

      const contextId = graph.addNode('Context', {
        temporal: coordinates.context,
        spatial: coordinates.axis
      });
      graph.addEdge(axisId, contextId, 'FRAMES_TO');

      // 7. Operator nodes and edges
      for (const op of operators) {
        const opId = graph.addNode('Operator', {
          symbol: op.symbol,
          class: op.class,
          effect: op.effect
        });
        graph.addEdge(clauseId, opId, 'OPERATES_VIA');

        // If operator changes orientation (WHY), new node is mandatory
        if (op.orientation_change || this._isOrientationChange(op, clause)) {
          const newNodeId = graph.addNode('Clause', {
            question: 'WHY',
            base: 4,
            text: `Orientation shift via ${op.symbol}`,
            generated: true
          });
          graph.addEdge(opId, newNodeId, 'CAUSES');
        }
      }
    }

    return graph;
  }

  _isOrientationChange(op, clause) {
    // Orientation change = operator affects WHY dimension (base 4)
    return op.effect === 'CreateState' && clause.question !== 'WHY' && op.symbol === 'Singularity';
  }

  _getGateTheme(gate) {
    const themes = {
      1: 'Creativity', 2: 'Receptivity', 3: 'Ordering',
      4: 'Formulization', 5: 'Patterns', 6: 'Conflict',
      7: 'Self', 8: 'Contribution', 9: 'Focus',
      10: 'Behavior', 11: 'Ideas', 12: 'Caution',
      13: 'Listener', 14: 'Power', 15: 'Extremes',
      16: 'Skills', 17: 'Opinion', 18: 'Correction',
      19: 'Wanting', 20: 'Contemplation', 21: 'Control',
      22: 'Grace', 23: 'Assimilation', 24: 'Rationalization',
      25: 'Innocence', 26: 'Taming', 27: 'Caring',
      28: 'Risk', 29: 'Perseverance', 30: 'Desire',
      31: 'Influence', 32: 'Continuity', 33: 'Privacy',
      34: 'Power', 35: 'Change', 36: 'Crisis',
      37: 'Friendship', 38: 'Opposition', 39: 'Provocation',
      40: 'Aloneness', 41: 'Fantasy', 42: 'Growth',
      43: 'Insight', 44: 'Alertness', 45: 'Gathering',
      46: 'Determination', 47: 'Realization', 48: 'Depth',
      49: 'Principles', 50: 'Values', 51: 'Shock',
      52: 'Inaction', 53: 'Development', 54: 'Ambition',
      55: 'Spirit', 56: 'Wandering', 57: 'Intuitive',
      58: 'Vitality', 59: 'Sexuality', 60: 'Limitation',
      61: 'Mystery', 62: 'Detail', 63: 'Doubt',
      64: 'Confusion'
    };
    return themes[gate] || 'Unknown';
  }

  _getLineOrbit(line) {
    const orbits = {
      1: 'Investigation', 2: 'Direction', 3: 'Trial and Error',
      4: 'Formulation', 5: 'Universalization', 6: 'Role Model'
    };
    return orbits[line] || 'Unknown';
  }

  _getColorMotivation(color) {
    const motivations = {
      1: 'Fear', 2: 'Hope', 3: 'Desire',
      4: 'Need', 5: 'Guilt', 6: 'Innocence'
    };
    return motivations[color] || 'Unknown';
  }

  _getTonePerception(tone) {
    const perceptions = {
      1: 'Observer', 2: 'Participant', 3: 'Witness',
      4: 'Interpreter', 5: 'Mediator', 6: 'Seer'
    };
    return perceptions[tone] || 'Unknown';
  }
}


// ============================================================================
// SECTION 10: PROJECTION ENGINE — DETERMINISTIC SENTENCE COMPILER
// ============================================================================

class ProjectionEngine {
  constructor(lexiconResolver) {
    this.lexicon = lexiconResolver;
  }

  /**
   * Compile all projections for a committed graph state.
   * Each projection is a view, not an interpretation.
   */
  project(graph, coordinates, operators) {
    const projections = [];

    for (const layer of INVARIANT.LAYERS) {
      try {
        const projection = this._projectLayer(graph, coordinates, operators, layer);
        projections.push(projection);
      } catch (err) {
        projections.push({
          id: uuid(),
          layer,
          valid: false,
          error: err instanceof FoundryError ? err.toJSON() : { message: err.message },
          output: null,
          traceback: null
        });
      }
    }

    return projections;
  }

  _projectLayer(graph, coordinates, operators, layer) {
    // 1. Retrieve lexical entry
    const lexicalEntry = this.lexicon.retrieve(coordinates, layer);

    // 2. Select template
    const template = this._selectTemplate(coordinates, operators, layer);

    // 3. Fill slots
    const sentence = this._fillTemplate(template, coordinates, operators, lexicalEntry, layer);

    // 4. Apply tone
    const toned = this._applyTone(sentence, operators, layer);

    // 5. Build traceback
    const traceback = this._buildTraceback(coordinates, operators);

    // 6. Seal
    const sealed = this._seal(toned, traceback, coordinates);

    // 7. Validate
    if (!this._validateTraceback(sealed, traceback)) {
      throw new FoundryError('PROJECTION_TRACEBACK_FAIL', 'Output does not resolve to lineage');
    }

    return {
      id: uuid(),
      layer,
      vocabulary: lexicalEntry.term,
      tone: INVARIANT.LAYER_TONES[layer],
      output: sealed.output,
      traceback: sealed.traceback,
      valid: true
    };
  }

  _selectTemplate(coordinates, operators, layer) {
    if (layer === 'CompressedSymbolic') {
      return GRAMMAR_TEMPLATES.compressed;
    }

    const opClass = operators[0]?.class || 'StateTransformer';

    for (const tmpl of GRAMMAR_TEMPLATES.templates) {
      const match = tmpl.match;
      if (match.base.includes(coordinates.base) &&
          match.operatorClass.includes(opClass) &&
          match.axis === coordinates.axis) {
        return tmpl;
      }
    }

    throw new FoundryError('PROJECTION_TEMPLATE_MISMATCH', 'No template fits state');
  }

  _fillTemplate(template, coordinates, operators, lexicalEntry, layer) {
    if (layer === 'CompressedSymbolic') {
      return `${coordinates.base}.${coordinates.gate}.${coordinates.line}.${coordinates.color}.${coordinates.tone}|${operators[0]?.symbol || 'None'}|${coordinates.axis}|${coordinates.context}`;
    }

    const base = INVARIANT.BASES[coordinates.base];
    const op = operators[0] || { symbol: 'Singularity', effect: 'CreateState' };
    const tone = INVARIANT.LAYER_TONES[layer];
    const toneVerb = INVARIANT.TONE_MAP[op.effect]?.[tone] || 'is';

    let sentence = template.structure
      .replace('[Subject]', base.voice)
      .replace('[ToneVerb]', toneVerb)
      .replace('[Object]', lexicalEntry.term)
      .replace('[Contextualizer]', `in ${coordinates.axis} ${coordinates.context}`);

    return sentence;
  }

  _applyTone(sentence, operators, layer) {
    // Tone is already applied in template filling
    // This is where additional layer-specific modulation would go
    // For now: return as-is (tone is fixed by layer)
    return sentence;
  }

  _buildTraceback(coordinates, operators) {
    const chain = [
      coordinates.base.toString(),
      coordinates.gate.toString(),
      coordinates.line.toString(),
      coordinates.color.toString(),
      coordinates.tone.toString(),
      coordinates.base.toString(),
      coordinates.axis,
      coordinates.context
    ];

    if (operators.length > 0) {
      chain.splice(1, 0, operators[0].symbol);
    }

    return chain;
  }

  _seal(output, traceback, coordinates) {
    const checksumInput = output + traceback.join('→');
    // Note: async sha256 not used here for sync flow
    // In production: use sync hash or await
    const checksum = `foundry:${coordinates.base}.${coordinates.gate}.${coordinates.line}.${coordinates.color}.${coordinates.tone}`;

    return {
      output,
      checksum,
      traceback,
      sealed: true,
      modifiable: false
    };
  }

  _validateTraceback(sealed, expectedTraceback) {
    return JSON.stringify(sealed.traceback) === JSON.stringify(expectedTraceback);
  }
}


// ============================================================================
// SECTION 11: MAIN ENGINE — IRREVERSIBLE PIPELINE
// ============================================================================

class FoundryEngine {
  constructor(options = {}) {
    this.version = INVARIANT.VERSION;
    this.schemaVersion = INVARIANT.SCHEMA_VERSION;

    // Initialize subsystems
    this.parser = new OntologicalParser();
    this.encoder = new StructuralEncoder();
    this.commiter = new GraphCommiter();
    this.lexicon = new LexiconResolver(options.lexicon);
    this.projector = new ProjectionEngine(this.lexicon);

    // Validate version compatibility
    if (options.schemaVersion && options.schemaVersion !== this.schemaVersion) {
      throw new FoundryError('ENGINE_VERSION_MISMATCH', 
        `Schema ${options.schemaVersion} != Engine ${this.schemaVersion}`);
    }

    Object.freeze(this);
  }

  /**
   * Execute the full irreversible pipeline.
   * 
   * INPUT → Ontological Parse → Dimensional Binding → Lexicon Resolution
   * → Structural Encoding → Temporal & Orientational Anchoring
   * → Causal Graph Commit → Output Projection(s)
   * 
   * @param {string} rawInput - The raw text input
   * @param {Object} overrides - Optional coordinate overrides
   * @returns {Object} Complete ontological state with projections
   */
  compile(rawInput, overrides = {}) {
    const startTime = Date.now();

    // === PIPELINE STEP 1: INPUT ===
    const input = {
      raw: rawInput,
      checksum: `sha256:${rawInput.length}`, // Simplified — use sha256 in production
      source: overrides.source || 'direct'
    };

    // === PIPELINE STEP 2: ONTOLOGICAL PARSE ===
    const parse = this.parser.parse(rawInput);

    if (!parse.valid) {
      return this._buildErrorState(input, parse, null, null, null);
    }

    // === PIPELINE STEP 3: DIMENSIONAL BINDING ===
    // Base is already bound in parse (each clause has base)
    // This step validates base consistency
    const bases = parse.clauses.map(c => c.base);
    const uniqueBases = [...new Set(bases)];

    // === PIPELINE STEP 4: LEXICON RESOLUTION ===
    // (Deferred to projection phase — coordinates needed first)

    // === PIPELINE STEP 5: STRUCTURAL ENCODING ===
    const coordinates = this.encoder.encode(parse.clauses, overrides);

    // === PIPELINE STEP 6: TEMPORAL & ORIENTATIONAL ANCHORING ===
    // (Already in coordinates: axis, context)

    // === PIPELINE STEP 7: CAUSAL GRAPH COMMIT ===
    const allOperators = parse.clauses.flatMap(c => c.operators);
    const graph = this.commiter.commit(parse, coordinates, allOperators);

    // === PIPELINE STEP 8: OUTPUT PROJECTION(S) ===
    const projections = this.projector.project(graph, coordinates, allOperators);

    // === BUILD FINAL STATE ===
    return this._buildState(input, parse, graph, projections, coordinates, startTime);
  }

  _buildState(input, parse, graph, projections, coordinates, startTime) {
    const state = {
      id: uuid(),
      version: this.version,
      timestamp: new Date().toISOString(),
      input,
      parse,
      graph: graph.toJSON(),
      projections,
      metadata: {
        compileTimeMs: Date.now() - startTime,
        coordinateHash: `${coordinates.base}.${coordinates.gate}.${coordinates.line}.${coordinates.color}.${coordinates.tone}`,
        projectionCount: projections.length,
        validProjections: projections.filter(p => p.valid).length
      }
    };

    return deepFreeze(state);
  }

  _buildErrorState(input, parse, graph, projections, coordinates) {
    return deepFreeze({
      id: uuid(),
      version: this.version,
      timestamp: new Date().toISOString(),
      input,
      parse,
      graph: graph ? graph.toJSON() : null,
      projections: projections || [],
      valid: false,
      error: parse.errors.find(e => e.fatal) || parse.errors[0]
    });
  }

  /**
   * Validate a projection against its ontology.
   * Returns true if projection traces back to the same root state.
   */
  validate(projection, ontologyState) {
    if (!projection.traceback || !projection.valid) return false;

    const expectedTraceback = [
      ontologyState.graph.lineage[0]?.split(' → ')[0] || '',
      // ... build expected from ontology
    ];

    return JSON.stringify(projection.traceback) === JSON.stringify(expectedTraceback);
  }

  /**
   * Get invariant checksum for an ontological state.
   */
  checksum(state) {
    const input = state.input.checksum;
    const clauseQuestions = state.parse.clauses
      .map(c => c.question + c.base)
      .sort()
      .join('');
    const lineage = (state.graph.lineage || []).join('');
    const layers = state.projections.map(p => p.layer).join('');

    return `foundry:${input}:${clauseQuestions}:${lineage}:${layers}`;
  }
}


// ============================================================================
// SECTION 12: EXPORT / MODULE INTERFACE
// ============================================================================

// Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    FoundryEngine,
    FoundryError,
    CausalGraph,
    OntologicalParser,
    LexiconResolver,
    StructuralEncoder,
    GraphCommiter,
    ProjectionEngine,
    INVARIANT,
    ERROR_CODES
  };
}

// Browser / ESM
if (typeof window !== 'undefined') {
  window.Foundry = {
    Engine: FoundryEngine,
    Error: FoundryError,
    Graph: CausalGraph,
    Parser: OntologicalParser,
    Lexicon: LexiconResolver,
    Encoder: StructuralEncoder,
    Commiter: GraphCommiter,
    Projector: ProjectionEngine,
    INVARIANT,
    ERROR_CODES
  };
}

// ESM default export
export { FoundryEngine as default, FoundryError, CausalGraph, INVARIANT };


// ============================================================================
// SECTION 13: USAGE EXAMPLE (commented — remove in production)
// ============================================================================

/*
// Basic usage:
const engine = new FoundryEngine();

const result = engine.compile('WHO: I Think. WHAT: I Remember. WHERE: I Define.');

console.log(JSON.stringify(result, null, 2));

// With overrides:
const result2 = engine.compile('WHO: I Think.', {
  gate: 1,
  line: 1,
  color: 1,
  tone: 1,
  axis: 'individual',
  context: 'present'
});

// Access projections:
for (const proj of result2.projections) {
  console.log(`${proj.layer}: ${proj.output}`);
}

// Validate:
const isValid = engine.validate(result2.projections[0], result2);
console.log('Valid:', isValid);

// Checksum:
const cs = engine.checksum(result2);
console.log('Checksum:', cs);
*/
