import {resolveSyntaxMarks} from './source-syntax.mjs';
// Fixed slot geometry transcribed from the supplied T23 generative-slot grammar.
export const SLOT_GEOMETRY = Object.freeze(['dimension','zodiac','gate','line','planetary','color','tone','base','center','biology','house','axis','yijing']);
const canonical = value => JSON.stringify(value, (_, v) => v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])) : v);

export class LexicalFillEngine {
  constructor({memory = [], persist = () => {}} = {}) {
    this.memory = new Map(memory.map(record=>[record.key,structuredClone(record)]));
    this.persist = persist;
  }
  resolve({lexemes, deterministic = {}, rules = [], punctuationMarks = [], axis = null, rulebookVersion = '1'}) {
    for (const item of lexemes) {
      if (!SLOT_GEOMETRY.includes(item.slot) || typeof item.word !== 'string' || !item.word.trim()) throw Error('Lexemes need a known slot and a word.');
    }
    const positioned = SLOT_GEOMETRY.flatMap(slot=>lexemes.filter(x=>x.slot===slot).map(x=>structuredClone(x)));
    const slots = structuredClone(deterministic), trace = [];
    const syntaxOperators = resolveSyntaxMarks(punctuationMarks);
    const context = {positioned, deterministic, punctuationMarks, axis, rulebookVersion};
    const key = canonical(context);
    const prior = this.memory.get(key);
    if (prior) return structuredClone({...prior.result, replayed:true});
    for (const rule of rules) {
      if (!rule.id || !SLOT_GEOMETRY.includes(rule.slot)) throw Error('Inference rule needs an ID and known output slot.');
      if (Object.hasOwn(slots,rule.slot)) continue;
      if ((rule.when ?? []).every(condition => positioned.some(item=>item.slot===condition.slot && item.word.toLowerCase()===condition.word.toLowerCase())) &&
          (!rule.mark || punctuationMarks.includes(rule.mark)) && (!rule.axis || rule.axis===axis)) {
        const competing = rules.filter(other=>other.slot===rule.slot && other.id!==rule.id &&
          (!other.mark || punctuationMarks.includes(other.mark)) && (!other.axis || other.axis===axis) &&
          (other.when??[]).every(c=>positioned.some(x=>x.slot===c.slot&&x.word.toLowerCase()===c.word.toLowerCase())));
        if (competing.some(other=>canonical(other.value)!==canonical(rule.value))) continue;
        slots[rule.slot]=structuredClone(rule.value);
        trace.push({slot:rule.slot,ruleId:rule.id,evidence:structuredClone(rule.when??[]),source:rule.source??null});
      }
    }
    const result={positioned,slots,trace,syntaxOperators,axis,rulebookVersion,replayed:false,phase:'semantic-structure',unresolved:SLOT_GEOMETRY.filter(slot=>!Object.hasOwn(slots,slot))};
    // Persist successfully bounded inference, never cache an empty attempt as fixed.
    if (trace.length) {this.memory.set(key,{key,result:structuredClone(result)});this.persist(this.exportMemory());}
    return result;
  }
  refine(structure, refinement) {
    if (!structure.positioned.length) throw Error('Degree refinement requires lexical structure first.');
    const ranges={degree:29,minute:59,second:59,arc:99};
    for (const [field,value] of Object.entries(refinement)) if (!(field in ranges)||!Number.isInteger(value)||value<0||value>ranges[field]) throw Error(`Invalid refinement: ${field}`);
    return structuredClone({...structure,phase:'refinement',refinement});
  }
  exportMemory(){return structuredClone([...this.memory.values()]);}
}
