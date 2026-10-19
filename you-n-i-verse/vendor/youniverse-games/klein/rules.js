// klein/rules.js — reversible production rules (Klein's "metamorphosis"
// insight: ONE rule definition drives both generation and recognition).
// Rules are DATA (JSON-serializable); a RuleSet is a per-world grammar, and
// world-knowledge constraints ("guardians block the unproven") are written in
// the SAME notation as grammar rules because relations are logical operators.
//
// Rule shapes:
//   rewrite rule: {id, kind:'rewrite', surface:'template with {slots}',
//                  slots: {name: {kind:'unit'|'list'|'vocab', vocab?}}}
//   logical rule: {id, kind:'logical', when:{subjectRequire?, subjectForbid?,
//                  objectRequire?, objectForbid?}, then:{relation, setObject?,
//                  effect}} — a relation applied as a Boolean operator over
//                  unit feature vectors.
//
// Errors throw; nothing is silent.

import { hasFeature, serializeUnit, deserializeUnit } from './units.js';

function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

export class RuleSet {
  /**
   * @param {string} name - grammar/world name
   * @param {Array} rules - plain-data rule definitions (validated + cloned)
   */
  constructor(name, rules) {
    if (!name) throw new Error('klein.RuleSet: name required');
    if (!Array.isArray(rules) || rules.length === 0) throw new Error('klein.RuleSet: non-empty rules array required');
    this.name = name;
    // enforce serializability: rules are DATA
    this.rules = JSON.parse(JSON.stringify(rules));
    for (const r of this.rules) {
      if (!r.id || (r.kind !== 'rewrite' && r.kind !== 'logical')) {
        throw new Error(`klein.RuleSet "${name}": each rule needs id and kind rewrite|logical`);
      }
      if (r.kind === 'rewrite' && (typeof r.surface !== 'string' || !r.slots)) {
        throw new Error(`klein.RuleSet "${name}": rewrite rule "${r.id}" needs surface template + slots`);
      }
    }
  }

  get(id) {
    const r = this.rules.find((x) => x.id === id);
    if (!r) throw new Error(`klein.RuleSet "${this.name}": unknown rule "${id}"`);
    return r;
  }

  /**
   * GENERATE: semantic structure -> surface form.
   * @param {string} ruleId
   * @param {object} bindings - slot -> unit | unit[] | string | string[]
   * @returns {string} surface text
   */
  generate(ruleId, bindings) {
    const rule = this.get(ruleId);
    if (rule.kind !== 'rewrite') throw new Error(`klein.RuleSet.generate: "${ruleId}" is not a rewrite rule`);
    let out = rule.surface;
    for (const [slot, spec] of Object.entries(rule.slots)) {
      const v = bindings[slot];
      if (v === undefined) throw new Error(`klein.RuleSet.generate: missing binding for slot "${slot}" in rule "${ruleId}"`);
      const lex = (x) => (typeof x === 'string' ? x : x.name);
      const text = spec.kind === 'list'
        ? (Array.isArray(v) ? v : [v]).map(lex).join(', ')
        : lex(v);
      out = out.replaceAll(`{${slot}}`, text);
    }
    return out;
  }

  /**
   * RECOGNIZE: surface form -> semantic structure (the SAME rules run backward).
   * Slots bounded by literal template delimiters are captured and resolved:
   * unit/list slots map names -> registered units via the registry;
   * vocab slots validate membership. Free-form-ish: tolerant of extra
   * whitespace and case differences.
   * @param {string} surface
   * @param {object} opts - {registry} required when any slot kind is unit|list
   * @returns {{ruleId, bindings}|null} null when no rule matches
   */
  recognize(surface, { registry } = {}) {
    if (typeof surface !== 'string' || !surface.trim()) throw new Error('klein.RuleSet.recognize: surface string required');
    const text = surface.trim().replace(/\s+/g, ' ');
    for (const rule of this.rules) {
      if (rule.kind !== 'rewrite') continue;
      const slotNames = [];
      const pattern = rule.surface.replace(/\{(\w+)\}/g, (m, s) => {
        slotNames.push(s);
        return '___SLOT___';
      });
      const parts = pattern.split('___SLOT___').map(escapeRe);
      const regex = new RegExp('^\\s*' + parts.join('\\s*(.+?)\\s*') + '\\s*$', 'i');
      const m = text.match(regex);
      if (!m) continue;
      const bindings = {};
      let ok = true;
      slotNames.forEach((slot, i) => {
        if (!ok) return;
        const spec = rule.slots[slot];
        const raw = m[i + 1].trim();
        if (spec.kind === 'list') {
          const items = raw.split(',').map((s) => s.trim()).filter(Boolean);
          if (spec.vocab) {
            if (!items.every((x) => spec.vocab.includes(x.toLowerCase()))) { ok = false; return; }
            bindings[slot] = items.map((x) => x.toLowerCase());
          } else {
            bindings[slot] = registry ? items.map((x) => registry.get(x)) : items;
          }
        } else if (spec.kind === 'vocab') {
          const hit = spec.vocab.find((v) => v.toLowerCase() === raw.toLowerCase());
          if (!hit) { ok = false; return; }
          bindings[slot] = hit;
        } else { // unit
          if (!registry) { ok = false; return; }
          bindings[slot] = registry.get(raw);
        }
      });
      if (ok) return { ruleId: rule.id, bindings };
    }
    return null;
  }

  /**
   * Evaluate a logical-operator rule (world knowledge as grammar).
   * @returns {{ruleId, relation, effect, setObject?, subject, object}|null}
   */
  evaluate(ruleId, subject, object) {
    const rule = this.get(ruleId);
    if (rule.kind !== 'logical') throw new Error(`klein.RuleSet.evaluate: "${ruleId}" is not a logical rule`);
    const check = (unit, require = [], forbid = []) =>
      require.every((f) => hasFeature(unit, f)) && forbid.every((f) => !hasFeature(unit, f));
    const w = rule.when ?? {};
    if (!check(subject, w.subjectRequire, w.subjectForbid)) return null;
    if (object && !check(object, w.objectRequire, w.objectForbid)) return null;
    if (!object && (w.objectRequire?.length || w.objectForbid?.length)) return null;
    return { ruleId: rule.id, relation: rule.then.relation, effect: rule.then.effect ?? null, setObject: rule.then.setObject ?? [], subject: subject.name, object: object?.name ?? null };
  }

  /** Rules are data: full JSON round-trip. */
  toJSON() { return { name: this.name, rules: this.rules }; }
  static fromJSON(data) {
    if (!data?.name || !Array.isArray(data.rules)) throw new Error('klein.RuleSet.fromJSON: {name, rules[]} required');
    return new RuleSet(data.name, data.rules);
  }
}

// --- world-knowledge rule library (same notation as grammar) ---------------
export const WORLD_KNOWLEDGE_RULES = [
  {
    id: 'guardians-block-the-unproven', kind: 'logical',
    when: { subjectRequire: ['ROLE_GUARDIAN', 'AGENT'], objectForbid: ['PROVEN'] },
    then: { relation: 'blocks', effect: 'deny-passage', setObject: ['BLOCKED'] },
  },
  {
    id: 'guardians-yield-to-the-proven', kind: 'logical',
    when: { subjectRequire: ['ROLE_GUARDIAN'], objectRequire: ['PROVEN'] },
    then: { relation: 'admits', effect: 'open-the-way' },
  },
];

// Serialization helpers kept here so rules can carry units inside bindings.
export { serializeUnit, deserializeUnit };
