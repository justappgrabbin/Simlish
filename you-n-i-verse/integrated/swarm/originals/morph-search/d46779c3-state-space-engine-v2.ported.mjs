/**
 * GENERATED FILE — mechanical TS->JS type-erasure transpile (esbuild transform, no logic changes).
 * Source: state-space-engine-v2.ts (canonical original in this directory)
 * Generated: 2026-09-22, esbuild 0.27.7. Adapter change: relative import specifiers rewritten to generated .ported.mjs filenames.
 */
const MovementOperator = {
  name: "Movement",
  keynote: "I Define",
  element: "Fire",
  apply(node, state) {
    const impulseFrequency = node.gate / 64;
    const impulseWaveform = Math.sin(node.line * Math.PI / 6);
    return {
      ...node,
      amplitude: impulseFrequency * 0.5 + 0.5,
      // Map to 0.5-1.0
      phase: impulseWaveform * Math.PI,
      dimension: "Movement",
      children: []
    };
  }
};
const EvolutionOperator = {
  name: "Evolution",
  keynote: "I Remember",
  element: "Water",
  apply(node, state) {
    const polarityRatio = node.color / 6;
    const polarityFrequency = node.tone / 6;
    const exaltation = node.amplitude * polarityRatio;
    const detriment = node.amplitude * (1 - polarityRatio);
    const evolvedAmplitude = Math.sqrt(exaltation ** 2 + detriment ** 2 + 2 * exaltation * detriment * Math.cos(polarityFrequency * Math.PI));
    return {
      ...node,
      amplitude: Math.min(1, evolvedAmplitude),
      phase: node.phase + polarityFrequency * Math.PI,
      dimension: "Evolution",
      children: [{
        ...node,
        id: `${node.id}_exaltation`,
        amplitude: exaltation,
        phase: node.phase,
        dimension: "Movement",
        parent: node,
        children: []
      }, {
        ...node,
        id: `${node.id}_detriment`,
        amplitude: detriment,
        phase: node.phase + Math.PI,
        dimension: "Movement",
        parent: node,
        children: []
      }]
    };
  }
};
const BeingOperator = {
  name: "Being",
  keynote: "I Am",
  element: "Earth",
  apply(node, state) {
    const witnessPerspective = node.base / 6;
    const selfInterference = node.amplitude * Math.cos(node.phase);
    const witnessNode = {
      ...node,
      id: `${node.id}_witness`,
      amplitude: Math.abs(selfInterference),
      phase: node.phase + Math.PI / 2,
      // Quadrature phase
      dimension: "Being",
      parent: node,
      children: []
    };
    state.edges.push({
      from: witnessNode.id,
      to: node.id,
      weight: witnessPerspective,
      type: "feedback",
      coherence: Math.abs(selfInterference)
    });
    return {
      ...node,
      amplitude: Math.abs(selfInterference),
      phase: node.phase * witnessPerspective,
      dimension: "Being",
      children: [witnessNode, ...node.children]
    };
  }
};
const DesignOperator = {
  name: "Design",
  keynote: "I Design",
  element: "Air",
  apply(node, state) {
    const harmonicGate = (node.gate + 31) % 64 || 64;
    const harmonicPartners = Array.from(state.nodes.values()).filter((n) => n.gate === harmonicGate && n.id !== node.id);
    for (const partner of harmonicPartners) {
      const edgeWeight = node.amplitude * partner.amplitude * Math.cos(node.phase - partner.phase);
      state.edges.push({
        from: node.id,
        to: partner.id,
        weight: edgeWeight,
        type: "harmonic",
        coherence: Math.abs(edgeWeight)
      });
    }
    const zodiacSigns = [
      "Aries",
      "Taurus",
      "Gemini",
      "Cancer",
      "Leo",
      "Virgo",
      "Libra",
      "Scorpio",
      "Sagittarius",
      "Capricorn",
      "Aquarius",
      "Pisces"
    ];
    const zodiacIndex = Math.floor(node.phase / (2 * Math.PI) * 12) % 12;
    const house = node.line;
    const planets = [
      "Sun",
      "Moon",
      "Mercury",
      "Venus",
      "Mars",
      "Jupiter",
      "Saturn",
      "Uranus",
      "Neptune",
      "Pluto",
      "North Node",
      "South Node",
      "Earth"
    ];
    const planetIndex = (node.gate - 1) % 13;
    return {
      ...node,
      zodiac: zodiacSigns[zodiacIndex],
      house,
      planet: planets[planetIndex],
      degree: node.phase / (2 * Math.PI) * 360,
      dimension: "Design",
      children: [...node.children]
    };
  }
};
const SpaceOperator = {
  name: "Space",
  keynote: "I Think",
  element: "Ether",
  apply(node, state) {
    const connectedEdges = state.edges.filter((e) => e.from === node.id || e.to === node.id);
    let emergentMeaning = 0;
    let totalWeight = 0;
    for (const edge of connectedEdges) {
      emergentMeaning += edge.coherence * edge.weight;
      totalWeight += Math.abs(edge.weight);
    }
    const normalizedMeaning = totalWeight > 0 ? emergentMeaning / totalWeight : 0;
    const finalAmplitude = Math.abs(normalizedMeaning) * node.amplitude;
    const meaningNode = {
      ...node,
      id: `${node.id}_meaning`,
      amplitude: finalAmplitude,
      phase: 0,
      // Meaning has no phase — it's the scalar closure
      dimension: "Space",
      parent: node,
      children: []
    };
    return {
      ...node,
      amplitude: finalAmplitude,
      phase: 0,
      dimension: "Space",
      children: [meaningNode, ...node.children]
    };
  }
};
class StateSpaceEngine {
  state;
  operators;
  nodeCounter;
  constructor() {
    this.nodeCounter = 0;
    this.operators = {
      Movement: MovementOperator,
      Evolution: EvolutionOperator,
      Being: BeingOperator,
      Design: DesignOperator,
      Space: SpaceOperator
    };
    this.state = {
      nodes: /* @__PURE__ */ new Map(),
      edges: [],
      dimensions: this.operators,
      coherence: 0,
      timestamp: Date.now(),
      history: []
    };
  }
  /**
   * Create a new node from raw coordinates.
   * Coordinates: Gate.Line.Color.Tone.Base
   */
  createNode(gate, line, color, tone, base, degree = 0, minute = 0, second = 0, arcSecond = 0) {
    const id = `node_${++this.nodeCounter}`;
    return {
      id,
      gate: Math.max(1, Math.min(64, gate)),
      line: Math.max(1, Math.min(6, line)),
      color: Math.max(1, Math.min(6, color)),
      tone: Math.max(1, Math.min(6, tone)),
      base: Math.max(1, Math.min(5, base)),
      // FIXED: 5 bases (1-5), was clamped to 6
      degree: Math.max(0, Math.min(360, degree)),
      minute: Math.max(0, Math.min(60, minute)),
      second: Math.max(0, Math.min(60, second)),
      // FIXED: was `Math.min(99, second)` with undefined `second` (ReferenceError) + duplicate key
      arcSecond: Math.max(0, Math.min(99, arcSecond)),
      axis: "",
      zodiac: "",
      house: 0,
      planet: "",
      dimension: "Movement",
      amplitude: 0,
      phase: 0,
      children: [],
      parent: null,
      timestamp: Date.now()
    };
  }
  /**
   * Parse coordinate string: "Gate.Line.Color.Tone.Base" or "25.1.3.5.2"
   */
  parseCoordinate(coord) {
    const degMatch = coord.match(/°(\d+)/);
    const minMatch = coord.match(/′(\d+)/);
    const secMatch = coord.match(/″(\d+)/);
    const arcSecMatch = coord.match(/‴(\d+)/);
    const degree = degMatch ? parseInt(degMatch[1]) : 0;
    const minute = minMatch ? parseInt(minMatch[1]) : 0;
    const second = secMatch ? parseInt(secMatch[1]) : 0;
    const arcSecond = arcSecMatch ? parseInt(arcSecMatch[1]) : 0;
    const cleanCoord = coord.replace(/°\d+/, "").replace(/′\d+/, "").replace(/″\d+/, "").replace(/‴\d+/, "");
    const parts = cleanCoord.split(/[.\s]+/).map(Number).filter((n) => !isNaN(n));
    return this.createNode(
      parts[0] || 1,
      parts[1] || 1,
      parts[2] || 1,
      parts[3] || 1,
      parts[4] || 1,
      degree,
      minute,
      second,
      arcSecond
    );
  }
  /**
   * Apply the FULL dimensional pipeline to a node.
   * Movement → Evolution → Being → Design → Space
   */
  applyPipeline(node) {
    this.saveSnapshot();
    let current = this.operators.Movement.apply(node, this.state);
    this.state.nodes.set(current.id, current);
    current = this.operators.Evolution.apply(current, this.state);
    this.state.nodes.set(current.id, current);
    for (const child of current.children) {
      this.state.nodes.set(child.id, child);
    }
    current = this.operators.Being.apply(current, this.state);
    this.state.nodes.set(current.id, current);
    for (const child of current.children) {
      this.state.nodes.set(child.id, child);
    }
    current = this.operators.Design.apply(current, this.state);
    this.state.nodes.set(current.id, current);
    current = this.operators.Space.apply(current, this.state);
    this.state.nodes.set(current.id, current);
    for (const child of current.children) {
      this.state.nodes.set(child.id, child);
    }
    this.state.coherence = this.calculateCoherence();
    this.state.timestamp = Date.now();
    return current;
  }
  /**
   * Apply a SINGLE dimension operator to a node.
   */
  applyDimension(node, dimension) {
    this.saveSnapshot();
    const operator = this.operators[dimension];
    const result = operator.apply(node, this.state);
    this.state.nodes.set(result.id, result);
    this.state.coherence = this.calculateCoherence();
    this.state.timestamp = Date.now();
    return result;
  }
  /**
   * Calculate global coherence from the graph structure.
   * Coherence = average edge weight magnitude, weighted by edge coherence.
   */
  calculateCoherence() {
    if (this.state.edges.length === 0) return 0;
    let totalCoherence = 0;
    let totalWeight = 0;
    for (const edge of this.state.edges) {
      totalCoherence += edge.coherence * Math.abs(edge.weight);
      totalWeight += Math.abs(edge.weight);
    }
    return totalWeight > 0 ? totalCoherence / totalWeight : 0;
  }
  /**
   * Propagate state through time.
   * Each timestep: all nodes evolve, edges update, new edges may form.
   */
  step() {
    this.saveSnapshot();
    for (const node of this.state.nodes.values()) {
      node.amplitude *= 0.95;
      const incomingEdges = this.state.edges.filter((e) => e.to === node.id);
      for (const edge of incomingEdges) {
        node.amplitude += edge.weight * edge.coherence * 0.1;
      }
      node.amplitude = Math.max(0, Math.min(1, node.amplitude));
      node.phase += 0.01 * node.amplitude;
      node.phase = node.phase % (2 * Math.PI);
    }
    for (const edge of this.state.edges) {
      const fromNode = this.state.nodes.get(edge.from);
      const toNode = this.state.nodes.get(edge.to);
      if (fromNode && toNode) {
        edge.coherence = Math.abs(Math.cos(fromNode.phase - toNode.phase));
        edge.weight = fromNode.amplitude * toNode.amplitude * edge.coherence;
      }
    }
    this.state.edges = this.state.edges.filter((e) => Math.abs(e.weight) > 0.01);
    this.state.coherence = this.calculateCoherence();
    this.state.timestamp = Date.now();
    return this.state;
  }
  /**
   * Run simulation for N timesteps.
   */
  simulate(steps) {
    const snapshots = [];
    for (let i = 0; i < steps; i++) {
      this.step();
      snapshots.push({
        nodes: Array.from(this.state.nodes.values()),
        edges: [...this.state.edges],
        coherence: this.state.coherence,
        timestamp: this.state.timestamp
      });
    }
    return snapshots;
  }
  /**
   * Query the state space.
   * Returns the most coherent node matching the query.
   */
  query(gate, dimension) {
    const candidates = Array.from(this.state.nodes.values()).filter((n) => (!gate || n.gate === gate) && (!dimension || n.dimension === dimension)).sort((a, b) => b.amplitude - a.amplitude);
    return candidates[0] || null;
  }
  /**
   * Generate sentence from node.
   */
  generateSentence(node) {
    const gateNames = [
      "The Creative",
      "The Receptive",
      "Difficulty at the Beginning",
      "Youthful Folly",
      "Waiting",
      "Conflict",
      "The Army",
      "Holding Together",
      "The Taming Power of the Small",
      "Treading",
      "Peace",
      "Standstill",
      "Fellowship with Men",
      "Possession in Great Measure",
      "Modesty",
      "Enthusiasm",
      "Following",
      "Work on What Has Been Spoiled",
      "Approach",
      "Contemplation",
      "Biting Through",
      "Grace",
      "Splitting Apart",
      "Return",
      "Innocence",
      "The Taming Power of the Great",
      "The Corners of the Mouth",
      "Preponderance of the Great",
      "The Abysmal",
      "The Clinging Fire",
      "Influence",
      "Duration",
      "Retreat",
      "The Power of the Great",
      "Progress",
      "Darkening of the Light",
      "The Family",
      "Opposition",
      "Obstruction",
      "Deliverance",
      "Decrease",
      "Increase",
      "Breakthrough",
      "Coming to Meet",
      "Gathering Together",
      "Pushing Upward",
      "Oppression",
      "The Well",
      "Revolution",
      "The Cauldron",
      "The Arousing",
      "Keeping Still",
      "Development",
      "The Marrying Maiden",
      "Abundance",
      "The Wanderer",
      "The Gentle",
      "The Joyous",
      "Dispersion",
      "Limitation",
      "Inner Truth",
      "Preponderance of the Small",
      "After Completion",
      "Before Completion"
    ];
    const lineNames = ["Investigator", "Hermit", "Martyr", "Opportunist", "Heretic", "Role Model"];
    const colorNames = ["Fear", "Hope", "Desire", "Need", "Guilt", "Innocence"];
    const toneNames = ["Security", "Uncertainty", "Action", "Meditation", "Judgement", "Acceptance"];
    const baseNames = ["Individuality", "Mind", "Body", "Ego", "Personality", "Spirit"];
    const hexagram = gateNames[node.gate - 1] || "Unknown";
    const line = lineNames[node.line - 1] || "Unknown";
    const color = colorNames[node.color - 1] || "Unknown";
    const tone = toneNames[node.tone - 1] || "Unknown";
    const base = baseNames[node.base - 1] || "Unknown";
    const dimKeynotes = {
      Movement: "I Define",
      Evolution: "I Remember",
      Being: "I Am",
      Design: "I Design",
      Space: "I Think"
    };
    return `${dimKeynotes[node.dimension]} \u2014 Gate ${node.gate} ${hexagram} through ${line} expression. ${color} motivates, ${tone} resonates, ${base} anchors. At ${node.degree}\xB0${node.minute}\u2032${node.second}\u2033${node.arcSecond}\u2034${node.axis ? " " + node.axis : ""}. Coherence: ${node.amplitude.toFixed(3)}`;
  }
  /**
   * Get current state.
   */
  getState() {
    return this.state;
  }
  /**
   * Get state history.
   */
  getHistory() {
    return this.state.history;
  }
  saveSnapshot() {
    this.state.history.push({
      nodes: Array.from(this.state.nodes.values()).map((n) => ({ ...n, children: [] })),
      edges: [...this.state.edges],
      coherence: this.state.coherence,
      timestamp: Date.now()
    });
    if (this.state.history.length > 1e3) {
      this.state.history = this.state.history.slice(-1e3);
    }
  }
}
var state_space_engine_v2_default = StateSpaceEngine;
export {
  BeingOperator,
  DesignOperator,
  EvolutionOperator,
  MovementOperator,
  SpaceOperator,
  StateSpaceEngine,
  state_space_engine_v2_default as default
};
