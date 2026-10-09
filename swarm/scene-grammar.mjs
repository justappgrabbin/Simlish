// Pure Synthia Automata — merged from synthia-recursive-media-field-v1.9/src/srmf-media-grammar-recorder.mjs (SceneGrammarRule / SceneGrammar)

const clamp = (n, min = 0, max = 1) => Math.max(min, Math.min(max, Number(n) || 0));
const clone = value => (value == null ? value : JSON.parse(JSON.stringify(value)));

export const WEIGHT_MIN = 0.05;
export const WEIGHT_MAX = 8;

/**
 * One weighted grammar rule over a scene context
 * ({tick, entityCount, tension, ...}). Condition fields are all optional:
 * {minEntities, maxEntities, minTension, maxTension, tickModulo}.
 */
export class SceneGrammarRule {
  constructor({ id, type, weight = 1, condition = {}, action = {} }) {
    if (!id) throw new Error('SceneGrammarRule requires an id');
    this.id = id;
    this.type = type || 'generic';
    this.weight = clamp(Number(weight), WEIGHT_MIN, WEIGHT_MAX) || WEIGHT_MIN;
    this.condition = {
      minEntities: condition.minEntities,
      maxEntities: condition.maxEntities,
      minTension: condition.minTension,
      maxTension: condition.maxTension,
      tickModulo: condition.tickModulo,
    };
    this.action = clone(action) || {};
    this.uses = 0;
    this.success = { count: 0, mean: 0.5 };
  }

  /** True when every specified condition bound holds for the context. */
  matches(context = {}) {
    const c = this.condition;
    const entities = Number(context.entityCount ?? context.entities ?? 0);
    const tension = Number(context.tension ?? 0);
    const tick = Number(context.tick ?? 0);
    if (Number.isFinite(c.minEntities) && entities < c.minEntities) return false;
    if (Number.isFinite(c.maxEntities) && entities > c.maxEntities) return false;
    if (c.minTension != null && tension < c.minTension) return false;
    if (c.maxTension != null && tension > c.maxTension) return false;
    if (c.tickModulo && tick % c.tickModulo !== 0) return false;
    return true;
  }

  /**
   * Reinforcement: tracks an exponentially-weighted success mean and nudges
   * weight by (reward - 0.5) * 0.25, hard-clamped to [0.05, 8].
   */
  reinforce(reward, alpha = 0.18) {
    const score = clamp(Number(reward));
    const current = this.success;
    current.mean = current.count === 0 ? score : current.mean + alpha * (score - current.mean);
    current.count += 1;
    this.weight = clamp(this.weight + (score - 0.5) * 0.25, WEIGHT_MIN, WEIGHT_MAX);
    return { mean: current.mean, count: current.count, weight: this.weight };
  }

  /** Effective selection weight: base weight modulated by learned success. */
  effectiveWeight() {
    return this.weight * (0.6 + this.success.mean * 0.8);
  }

  snapshot() {
    return {
      id: this.id, type: this.type, weight: this.weight,
      condition: clone(this.condition), action: clone(this.action),
      uses: this.uses, success: { ...this.success },
    };
  }
}

/**
 * A small deterministic scene grammar. select(context) picks the matching rule
 * with the highest effective weight (ties broken by insertion order — no RNG).
 */
export class SceneGrammar {
  constructor() {
    this.rules = new Map();
    this.history = [];
  }

  addRule(spec) {
    if (this.rules.has(spec.id)) throw new Error(`Duplicate grammar rule: ${spec.id}`);
    const rule = new SceneGrammarRule(spec);
    this.rules.set(rule.id, rule);
    return rule;
  }

  /** Matching rules with their effective weights, best first. */
  eligibleRules(context = {}) {
    return [...this.rules.values()]
      .filter(rule => rule.matches(context))
      .map(rule => ({ rule, effectiveWeight: rule.effectiveWeight() }))
      .sort((a, b) => b.effectiveWeight - a.effectiveWeight);
  }

  /** Best matching rule for the context, or null when nothing matches. */
  select(context = {}) {
    const eligible = this.eligibleRules(context);
    if (!eligible.length) return null;
    const picked = eligible[0].rule;
    picked.uses += 1;
    this.history.push({ tick: Number(context.tick ?? 0), ruleId: picked.id, effectiveWeight: eligible[0].effectiveWeight });
    return picked;
  }

  /** Reinforce a rule by id; returns its post-update record, or null if unknown. */
  reinforce(id, reward) {
    const rule = this.rules.get(id);
    if (!rule) return null;
    return { id, ...rule.reinforce(reward) };
  }

  snapshot() {
    return { rules: [...this.rules.values()].map(r => r.snapshot()), history: [...this.history] };
  }
}

export default SceneGrammar;
