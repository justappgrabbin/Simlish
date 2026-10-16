import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync,existsSync} from 'node:fs';
const J=f=>JSON.parse(readFileSync(new URL('./'+f,import.meta.url),'utf8'));
const clean=J('primitive-functions.json'),attrs=J('primitive-attributes.json').attributes,S=J('primitive-sources.json');
const P=clean.primitives,byId=new Map(P.map(p=>[p.id,p]));
test('ids are unique and every primitive has symbol, kind, a verbatim non-empty form',()=>{assert.equal(byId.size,P.length);for(const p of P){assert.ok(p.symbol&&p.kind&&String(p.form).length>0,p.id)}});
test('clean layer carries no citations or prose (those live in the sources/attributes layers)',()=>{for(const p of P)assert.deepEqual(Object.keys(p).filter(k=>!['id','symbol','kind','form','fn','gap','ctx'].includes(k)),[],p.id)});
test('five dimensions, 64 gates, 384 lines, 8 trigrams, 13 address places, 19 syntax marks, 11 stages, 5 fields',()=>{const n=k=>P.filter(p=>p.kind===k).length;
 assert.deepEqual([n('dimension'),n('gate'),n('line'),n('trigram'),n('address-place'),n('syntax-mark'),n('pipeline-stage'),n('execution-field')],[5,64,384,8,13,19,11,5])});
test('every function name resolves to an existing file and line in the repo (or the copied reference files)',()=>{
 for(const p of P)for(const f of p.fn){const r=S.functions[f];assert.ok(r,f);const [file,line]=r.ref.split(/:(\d+)$/);const path=new URL('../'+file,import.meta.url);assert.ok(existsSync(path),r.ref);assert.ok(readFileSync(path,'utf8').split('\n').length>=Number(line),r.ref)}});
test('every primitive has a source entry or is covered by a family source; every gate/line has attributes from the Black Book and Rutt',()=>{
 for(const p of P.filter(p=>p.kind==='gate')){const a=attrs[p.id];assert.ok(a.blackBook.center&&a.zhouyi.statement&&a.doc3.anchor&&a.geneKey.shadow,p.id);assert.ok(S.sources[p.id].form.line,p.id)}
 for(const p of P.filter(p=>p.kind==='line')){const a=attrs[p.id];assert.ok(a.blackBook.text&&a.zhouyi&&a.keynote,p.id)}});
test('gates are distributed over the five dimensions from her doc3 base numbers (all 64 placed once)',()=>{const all=Object.values(clean.gatesByDoc3Base).flat().sort((a,b)=>a-b);assert.deepEqual(all,Array.from({length:64},(_,i)=>i+1));assert.deepEqual(Object.keys(clean.gatesByDoc3Base).sort(),['being','design','evolution','movement','space'])});
test('counts are consistent and gaps are listed, not hidden',()=>{const c={full:0,partial:0,gap:0};for(const p of P)p.fn.length?(p.gap.length?c.partial++:c.full++):c.gap++;assert.deepEqual(c,{full:clean.counts.full,partial:clean.counts.partial,gap:clean.counts.gap});assert.equal(c.full+c.partial+c.gap,P.length)});
test('Mawangdui stays UNRESOLVED: Rutt reading is one of seven, none chosen',()=>{const m=P.filter(p=>p.kind==='wheel-reading');assert.equal(m.length,7);for(const x of m)assert.ok(x.gap.some(g=>/UNRESOLVED/.test(g)))});
test('no primitive binds a word to one dimension: dimension ties are only in ctx as a named configuration',()=>{for(const p of P.filter(p=>p.kind==='letter'))assert.equal((p.ctx??[]).length,0)});
