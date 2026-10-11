import {LexicalFillEngine} from './lexical-fill.mjs';
// Scene performance stays reusable; its moment and appearance are separate data.
export const MOMENT_FIELDS = Object.freeze([
  'planetary', 'dimension', 'gate', 'line', 'color', 'tone', 'base',
  'degree', 'minute', 'second', 'arc', 'zodiac', 'house'
]);

export function createMomentRecipe({sceneId, description = '', addresses = [], connections = [], appearance = {}, addressSource = 'unresolved', lexicalFill = [], inferenceRules = [], inferenceMemory = [], punctuationMarks = [], axis = null, rulebookVersion = '1', deterministic = {}}) {
  if (typeof sceneId !== 'string' || !sceneId.trim()) throw new Error('A reusable scene ID is required.');
  const ranges = {gate:[1,64],line:[1,6],color:[1,6],tone:[1,6],base:[1,5],degree:[0,29],minute:[0,59],second:[0,59],arc:[0,99],house:[1,12]};
  const positions = addresses.map(address => {
    const position = Object.fromEntries(MOMENT_FIELDS.map(field => [field, address[field] ?? null]));
    for (const [field,[min,max]] of Object.entries(ranges)) {
      const value = position[field];
      if (value !== null && (!Number.isInteger(value) || value < min || value > max)) throw new Error(`Invalid ${field}: ${value}`);
    }
    return position;
  });
  const engine = new LexicalFillEngine({memory:inferenceMemory});
  const semanticStructures = positions.map(position => {
    const positionSlots = Object.fromEntries(Object.entries(position).filter(([key,value]) => !['degree','minute','second','arc'].includes(key) && value !== null));
    const structure = engine.resolve({lexemes:lexicalFill,deterministic:{...deterministic,...positionSlots},rules:inferenceRules,punctuationMarks,axis,rulebookVersion});
    const refinement = Object.fromEntries(['degree','minute','second','arc'].filter(key=>position[key]!==null).map(key=>[key,position[key]]));
    return lexicalFill.length && Object.keys(refinement).length ? engine.refine(structure,refinement) : structure;
  });
  if (!positions.length && lexicalFill.length) semanticStructures.push(engine.resolve({lexemes:lexicalFill,deterministic,rules:inferenceRules,punctuationMarks,axis,rulebookVersion}));
  return structuredClone({version:2, semanticStructures, inferenceMemory:engine.exportMemory(), sceneId, moment:{description, addressSource, addresses:positions, connections}, appearance,
    referenceLibrary:{gates:64,referencePositions:9,subdivisionRange:[12,16]},
    // Subdivisions address conditions; they do not imply independent combinations.
    unresolved:positions.length ? positions.flatMap((p,i)=>MOMENT_FIELDS.filter(f=>p[f]===null).map(f=>`${i}.${f}`)) : ['moment.addresses']});
}
