import test from 'node:test';
import assert from 'node:assert/strict';
import {LexicalFillEngine} from './lexical-fill.mjs';
import {resolveSyntaxMarks} from './source-syntax.mjs';
test('syntax follows the supplied master document, not a three-mode approximation',()=>{
 const [transition,fork]=resolveSyntaxMarks(['.',';']);
 assert.equal(transition.name,'Transitioner');assert.equal(transition.linguisticEquivalent,'then / so / thus');
 assert.equal(fork.name,'Fork');assert.equal(fork.source.driveId,'1x7xzzzaxg_SdhHM5uteiZ21kGsoRfdK5');
 assert.throws(()=>resolveSyntaxMarks(['?']));
});
const input={lexemes:[{slot:'house',word:'store'},{slot:'dimension',word:'movement'}],deterministic:{gate:1,line:2,dimension:'Movement'},rules:[{id:'fixture-context',slot:'house',when:[{slot:'house',word:'store'}],value:2,source:'test fixture only'}]};
test('position precedes surface order; deterministic values cannot be overridden',()=>{
 const result=new LexicalFillEngine().resolve({...input,rules:[...input.rules,{id:'override',slot:'gate',value:64}]});
 assert.equal(result.positioned[0].slot,'dimension');assert.equal(result.slots.gate,1);assert.equal(result.slots.house,2);
});
test('inference survives export and reload and retains justification',()=>{
 let saved;const engine=new LexicalFillEngine({persist:records=>saved=records});engine.resolve(input);
 const replay=new LexicalFillEngine({memory:saved}).resolve({...input,rules:[]});
 assert.equal(replay.replayed,true);assert.equal(replay.slots.house,2);assert.equal(replay.trace[0].ruleId,'fixture-context');
});
test('conflicting rules are not frozen; degree cannot generate semantic structure',()=>{
 const engine=new LexicalFillEngine();const result=engine.resolve({...input,rules:[...input.rules,{...input.rules[0],id:'conflict',value:3}]});
 assert.ok(result.unresolved.includes('house'));assert.equal(engine.exportMemory().length,0);
 assert.throws(()=>engine.refine(engine.resolve({lexemes:[],deterministic:{}}),{degree:10}));
 const refined=engine.refine(engine.resolve(input),{degree:29,minute:59,second:59,arc:99});
 assert.equal(refined.slots.gate,1);assert.equal(refined.refinement.arc,99);
 assert.throws(()=>engine.refine(refined,{degree:30}));
});
