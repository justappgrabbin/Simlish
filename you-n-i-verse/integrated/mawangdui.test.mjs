import test from 'node:test';import assert from 'node:assert/strict';
import {mawangduiVersions,differ,MAWANGDUI_UNRESOLVED} from './mawangdui-versions.mjs';
import {WHEELS} from './wheels.mjs';
import {readFileSync} from 'node:fs';import {fileURLToPath} from 'node:url';import {dirname,join} from 'node:path';
const B=readFileSync(join(dirname(fileURLToPath(import.meta.url)),'reference/hexagram_sequences.js.txt'),'utf8');
test('every distinct Mawangdui ordering found is kept side by side; each is a permutation of the 64 gates; none is chosen',async()=>{
 const v=await mawangduiVersions(B);
 for(const x of Object.values(v)){assert.equal(x.macros.length,64,x.id);assert.equal(new Set(x.macros).size,64,x.id+' is a permutation of the 64 hexagrams');assert.ok(x.kingWen.every(k=>k>=1&&k<=64&&k!==undefined))}
 assert.equal(MAWANGDUI_UNRESOLVED.status,'UNRESOLVED');
 assert.deepEqual(WHEELS.Evolution.order,v.A.macros,'wheels.mjs (unchanged) is version A');
 assert.equal(differ(v.C,v.E),0);assert.ok(differ(v.A,v.C)>0&&differ(v.C,v.D)>0&&differ(v.C,v.Bliteral)>0,'the versions really differ');
 // the king wen sequence was NOT substituted
 const kw=Array.from({length:64},(_,i)=>i+1);for(const x of Object.values(v))assert.notDeepEqual([...x.kingWen],kw,x.id+' is not King Wen order');
});
