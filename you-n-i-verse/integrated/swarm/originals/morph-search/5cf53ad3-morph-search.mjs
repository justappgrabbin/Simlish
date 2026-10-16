// Local bounded ABC / PSO / GA search. Lower finite fitness is better.
// Candidate vectors contain mutable embodiment controls, never identity/address.
export function createMorphSearch({ bounds, evaluate, initial, population = 12, random = Math.random }) {
  if (!Array.isArray(bounds) || !bounds.length || bounds.length > 128 ||
      bounds.some(([lo, hi]) => !Number.isFinite(lo) || !Number.isFinite(hi) || lo >= hi) ||
      !Number.isInteger(population) || population < 4 || population > 64 || typeof evaluate !== 'function')
    throw new TypeError('Invalid bounded morph search');
  const clamp = vector => bounds.map(([lo, hi], i) => Math.min(hi, Math.max(lo, vector[i])));
  const fresh = () => bounds.map(([lo, hi]) => lo + random() * (hi - lo));
  let evaluations = 0;
  function candidate(vector) {
    const position = clamp(vector);
    if (position.some(value => !Number.isFinite(value))) throw new TypeError('Invalid morph candidate');
    const score = evaluate([...position]);
    if (!Number.isFinite(score)) throw new TypeError('Morph fitness must be finite');
    evaluations++;
    return { position, score, trials: 0, velocity: position.map(() => 0), best: [...position], bestScore: score };
  }
  let swarm = Array.from({ length: population }, (_, i) => candidate(i === 0 && initial ? initial : fresh()));
  let best = swarm.reduce((a, b) => a.score <= b.score ? a : b);
  best = { position: [...best.position], score: best.score };
  const counts = { ABC: 0, PSO: 0, GA: 0 };
  function keep(c) { if (c.score < best.score) best = { position: [...c.position], score: c.score }; }
  function bee(index) {
    const source = swarm[index];
    let peer = Math.floor(random() * (population - 1));
    if (peer >= index) peer++;
    const axis = Math.floor(random() * bounds.length);
    const vector = [...source.position];
    vector[axis] += (random() * 2 - 1) * (vector[axis] - swarm[peer].position[axis]);
    const next = candidate(vector);
    if (next.score < source.score) { swarm[index] = next; keep(next); }
    else source.trials++;
  }
  function abc() {
    // Employed bees explore neighbours; onlookers sample by relative fitness.
    for (let i = 0; i < population; i++) bee(i);
    const minimum = Math.min(...swarm.map(c => c.score));
    const weights = swarm.map(c => 1 / (1 + c.score - minimum));
    const total = weights.reduce((a, b) => a + b, 0);
    for (let i = 0; i < population; i++) {
      let cursor = random() * total, selected = population - 1;
      for (let j = 0; j < population; j++) { cursor -= weights[j]; if (cursor <= 0) { selected = j; break; } }
      bee(selected);
    }
    // Scouts replace stalled sources, preserving the global best separately.
    for (let i = 0; i < population; i++) if (swarm[i].trials >= 8) { swarm[i] = candidate(fresh()); keep(swarm[i]); }
    counts.ABC++;
  }
  function pso() {
    for (let i = 0; i < population; i++) {
      const source = swarm[i];
      const velocity = source.position.map((value, axis) => {
        const range = (bounds[axis][1] - bounds[axis][0]) * .25;
        return Math.max(-range, Math.min(range, .65 * source.velocity[axis] +
          1.4 * random() * (source.best[axis] - value) + 1.4 * random() * (best.position[axis] - value)));
      });
      const next = candidate(source.position.map((value, axis) => value + velocity[axis]));
      next.velocity = velocity;
      if (source.bestScore < next.score) { next.best = [...source.best]; next.bestScore = source.bestScore; }
      swarm[i] = next; keep(next);
    }
    counts.PSO++;
  }
  function tournament() {
    const a = swarm[Math.floor(random() * population)], b = swarm[Math.floor(random() * population)];
    return a.score <= b.score ? a : b;
  }
  function ga() {
    const next = [candidate(best.position)];
    while (next.length < population) {
      const a = tournament(), b = tournament();
      const child = a.position.map((value, axis) => {
        const mix = random();
        const mutation = random() < .2 ? (random() * 2 - 1) * (bounds[axis][1] - bounds[axis][0]) * .12 : 0;
        return mix * value + (1 - mix) * b.position[axis] + mutation;
      });
      const c = candidate(child); next.push(c); keep(c);
    }
    swarm = next; counts.GA++;
  }
  return {
    step() { abc(); pso(); ga(); return this.snapshot(); },
    snapshot() { return { position: [...best.position], score: best.score, evaluations, algorithms: { ...counts } }; }
  };
}

export function solveMorph(options, rounds = 6) {
  if (!Number.isInteger(rounds) || rounds < 1 || rounds > 32) throw new RangeError('Morph rounds must be 1–32');
  const search = createMorphSearch(options);
  for (let i = 0; i < rounds; i++) search.step();
  return search.snapshot();
}

// These are explicit animation control objectives, not chart-derived physics.
// Five simultaneous fields contribute; Space observes the candidate's limits.
export function solveMotionMorph(motion = 'idle', previous = [5, 1, .45], random = Math.random) {
  const target = [motion === 'walk' ? 12 : 5, motion === 'sit' ? .65 : ['lift', 'reach'].includes(motion) ? 1.2 : 1, .45];
  const result = solveMorph({ bounds: [[2, 16], [.5, 1.3], [.1, .65]], initial: previous, random,
    evaluate: ([frequency, height, angle]) =>
      ((frequency - target[0]) / 14) ** 2 + // Movement: requested action cadence
      .01 * ((height - previous[1]) / .8) ** 2 + // Evolution: continuity cost
      (height - target[1]) ** 2 + // Being: requested embodiment pose
      (angle - target[2]) ** 2 + // Design: wing articulation
      Math.max(0, height * Math.sin(angle) - .8) ** 2 // Space: articulation envelope
  });
  return { ...result, fields: ['Movement', 'Evolution', 'Being', 'Design', 'Space'], target };
}

export function solveWorldMorph(pieces = [], random = Math.random) {
  const anchors = pieces.slice(0, 8).map(piece => {
    const angle = (piece.address.arc ?? piece.address.gate * 20250) / 1296000 * Math.PI * 2;
    return { id: piece.id, angle, scale: 1.6 + piece.address.line / 6 };
  });
  if (!anchors.length) return { anchors: [], score: 0, algorithms: { ABC: 0, PSO: 0, GA: 0 } };
  // Search angular offsets and radii. Original addresses still supply the
  // preferred positions; observer-space penalizes overlapping structures.
  const initial = anchors.flatMap(() => [0, 12]);
  const result = solveMorph({ initial, random, bounds: anchors.flatMap(() => [[-.45, .45], [9, 17]]),
    evaluate: vector => {
      let fitness = 0;
      const positions = anchors.map((anchor, i) => {
        const offset = vector[i * 2], radius = vector[i * 2 + 1];
        fitness += offset ** 2 * .2 + ((radius - 12) / 8) ** 2 * .1;
        return [Math.cos(anchor.angle + offset) * radius, Math.sin(anchor.angle + offset) * radius];
      });
      for (let i = 0; i < positions.length; i++) for (let j = i + 1; j < positions.length; j++) {
        const distance = Math.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]);
        fitness += Math.max(0, anchors[i].scale + anchors[j].scale + 1 - distance) ** 2;
      }
      return fitness;
    }
  });
  return { ...result, anchors: anchors.map((anchor, i) => {
    const angle = anchor.angle + result.position[i * 2], radius = result.position[i * 2 + 1];
    return { ...anchor, angle, position: [Math.cos(angle) * radius, 0, Math.sin(angle) * radius] };
  }) };
}
