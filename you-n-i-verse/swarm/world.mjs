import { SceneGrammar } from './scene-grammar.mjs';
const copy = x => structuredClone(x);
const finite = (v, name) => { if (!Number.isFinite(v)) throw new TypeError(name + ' must be finite'); return v; };
// Explicit simulation parameters, in world units. These are not dimensional
// or astronomical correspondences. Users can supply another parameter set.
export const FLAVORS = Object.freeze({
  grounded: { gravity: 9.81, damping: 0.4, stiffness: 18, restitution: 0.25 },
  buoyant: { gravity: -2, damping: 0.2, stiffness: 8, restitution: 0.6 },
  elastic: { gravity: 4, damping: 0.05, stiffness: 40, restitution: 0.85 }
});
export class SwarmWorld {
  constructor() {
    this.particles = new Map(); this.composites = new Map(); this.edges = [];
    this.history = []; this.phase = 'embodiment'; this.tick = 0;
    this.parameters = copy(FLAVORS.grounded);
    this.grammar = new SceneGrammar();
    this.grammar.addRule({ id: 'intact', condition: { maxTension: 0.999999 }, action: { state: 'bound' } });
    this.grammar.addRule({ id: 'rupture', condition: { minTension: 1 }, action: { state: 'separated' } });
  }
  record(operation, details) {
    const event = { id: this.history.length + 1, parent: this.history.at(-1)?.id ?? null,
      tick: this.tick, phase: this.phase, operation, ...copy(details) };
    this.history.push(event); return event;
  }
  addParticle({ id, x, y, mass = 1, bit = 0, address = null, qualities = {}, maxSpeed = 20 }) {
    if (!id || this.particles.has(id)) throw new Error('Unique particle id required');
    finite(x, 'x'); finite(y, 'y'); finite(mass, 'mass'); finite(maxSpeed, 'maxSpeed');
    if (mass <= 0 || maxSpeed <= 0 || ![0, 1].includes(bit)) throw new RangeError('Invalid particle constraints');
    const p = { id, x, y, vx: 0, vy: 0, mass, bit, address: copy(address),
      qualities: Object.freeze(copy(qualities)), maxSpeed };
    this.particles.set(id, p); this.record('primitive', { particle: p }); return p;
  }
  compose(id, members, operator = 'o_bundle') {
    if (this.composites.has(id) || this.particles.has(id)) throw new Error('Duplicate composition');
    if (!['o_bundle', 'o_sequence'].includes(operator) || !members.length ||
        members.some(x => !this.particles.has(x) && !this.composites.has(x))) throw new Error('Invalid production');
    const c = { id, operator, members: [...members], lineage: this.history.length + 1 };
    this.composites.set(id, c); this.record('composition', { composite: c }); return c;
  }
  connect(source, target, { breakStrain = 0.6, context = 'composition' } = {}) {
    const a = this.particles.get(source), b = this.particles.get(target);
    if (!a || !b || a === b) throw new Error('Edge requires distinct particles');
    finite(breakStrain, 'breakStrain'); if (breakStrain <= 0) throw new RangeError('breakStrain must be positive');
    const rest = Math.hypot(b.x - a.x, b.y - a.y);
    if (!rest) throw new Error('Zero length bond');
    const e = { source, relation: 'elastic-bond', target, rest, breakStrain, active: true,
      address: [copy(a.address), copy(b.address)], dimension: null,
      transformation: 'Hooke spring', context };
    e.provenance = this.record('relation', { edge: e }).id; this.edges.push(e); return e;
  }
  requestFlavor(nameOrParameters) {
    const p = typeof nameOrParameters === 'string' ? FLAVORS[nameOrParameters] : nameOrParameters;
    if (!p) throw new Error('Unknown flavor');
    for (const k of ['gravity', 'damping', 'stiffness', 'restitution']) finite(p[k], k);
    if (p.damping < 0 || p.stiffness < 0 || p.restitution < 0 || p.restitution > 1) throw new RangeError('Invalid physics parameters');
    const before = this.parameters; this.parameters = copy(p);
    this.record('user-flavor', { before, after: p }); return this;
  }
  setPhase(phase) {
    if (!['embodiment', 'task'].includes(phase)) throw new Error('Invalid phase');
    this.phase = phase; this.record('phase', { phase });
  }
  impulse(id, x, y) {
    if (this.phase !== 'embodiment') throw new Error('Physical interaction requires embodiment phase');
    finite(x, 'impulse.x'); finite(y, 'impulse.y');
    const p = this.particles.get(id); if (!p) throw new Error('Unknown particle');
    p.vx += x / p.mass; p.vy += y / p.mass;
    this.record('impulse', { id, impulse: [x, y], velocity: [p.vx, p.vy] });
  }
  step(dt = 1 / 120) {
    finite(dt, 'dt'); if (dt <= 0 || dt > 1 / 60) throw new RangeError('Use fixed substeps <= 1/60');
    if (this.phase !== 'embodiment') return; // Task search never runs physics concurrently.
    const { gravity, damping, stiffness, restitution } = this.parameters;
    const forces = new Map([...this.particles].map(([id, p]) => [id, [0, p.mass * gravity]]));
    for (const edge of this.edges.filter(e => e.active)) {
      const a = this.particles.get(edge.source), b = this.particles.get(edge.target);
      const dx = b.x - a.x, dy = b.y - a.y, length = Math.hypot(dx, dy);
      const strain = Math.abs(length - edge.rest) / edge.rest;
      const production = this.grammar.select({ tension: strain / edge.breakStrain });
      if (production.action.state === 'separated') {
        edge.active = false;
        this.record('bond-rupture', { source: a.id, target: b.id, strain,
          rule: production.id, provenance: edge.provenance }); continue;
      }
      if (!length) continue;
      const f = stiffness * (length - edge.rest);
      forces.get(a.id)[0] += f * dx / length; forces.get(a.id)[1] += f * dy / length;
      forces.get(b.id)[0] -= f * dx / length; forces.get(b.id)[1] -= f * dy / length;
    }
    for (const [id, p] of this.particles) {
      const [fx, fy] = forces.get(id);
      p.vx = (p.vx + fx / p.mass * dt) * Math.exp(-damping * dt);
      p.vy = (p.vy + fy / p.mass * dt) * Math.exp(-damping * dt);
      const speed = Math.hypot(p.vx, p.vy);
      if (speed > p.maxSpeed) { p.vx *= p.maxSpeed / speed; p.vy *= p.maxSpeed / speed; }
      p.x += p.vx * dt; p.y += p.vy * dt;
      for (const [axis, velocity] of [['x', 'vx'], ['y', 'vy']]) {
        if (p[axis] < 0 || p[axis] > 10) {
          const before = p[velocity]; p[axis] = Math.max(0, Math.min(10, p[axis]));
          p[velocity] = -before * restitution;
          this.record('boundary-collision', { id, axis, before, after: p[velocity] });
        }
      }
    }
    this.tick++;
  }
  snapshot() {
    return copy({ particles: [...this.particles.values()], composites: [...this.composites.values()],
      edges: this.edges, parameters: this.parameters, phase: this.phase, tick: this.tick, history: this.history });
  }
}
