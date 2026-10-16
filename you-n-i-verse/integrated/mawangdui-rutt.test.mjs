import test from 'node:test';import assert from 'node:assert/strict';
import {mawangduiRuttKingWen,mawangduiRuttMacros,RUTT_TABLE_10} from './mawangdui-rutt.mjs';
import {mawangduiVersions,differ} from './mawangdui-versions.mjs';
import {readFileSync} from 'node:fs';
test('Rutt rule reproduces Rutt Table 10 for all 64 positions',()=>{assert.deepEqual(mawangduiRuttKingWen(),[...RUTT_TABLE_10])});
test('Table 10 is a permutation of 1..64',()=>{assert.deepEqual([...RUTT_TABLE_10].sort((a,b)=>a-b),Array.from({length:64},(_,i)=>i+1))});
test('Rutt reading is kept NEXT TO versions A-E, none chosen',async()=>{
 const src=readFileSync(new URL('./reference/hexagram_sequences.js.txt',import.meta.url),'utf8'),v=await mawangduiVersions(src);
 const r={macros:mawangduiRuttMacros()};
 const report=Object.fromEntries(Object.entries(v).map(([k,x])=>[k,differ(r,x)]));
 assert.ok(Object.keys(report).length>=5);                        // compared against every existing version
 assert.ok(report.A>0&&report.C>0);                               // differs from A (doubled-first, family-order lower) and C: the lower-trigram rule is the open point
 console.log('positions differing from Rutt:',JSON.stringify(report));
});
