import test from 'node:test';
import assert from 'node:assert/strict';
import {projectThreeTwoOne,humanLowestLines,transitionThreeTwoOne} from '../src/three-two-one.mjs';
import {projectZhouyiSixLine,transitionZhouyiSixLine} from '../src/zhouyi-six-line.mjs';
import {createUnifiedSwarm,UnifiedSwarm} from '../src/unified-swarm.mjs';
import {fromLines} from '../vendor/pass05/zhouyi-state.mjs';

const levels=()=>({
  level3:[{bit:1},{bit:0},{bit:1}],
  level2:[{bit:0},{bit:1}],
  level1:[{bit:1}],
});
const human=()=>({color:4,tone:2,base:5,line:6,arcAxis:{axis:'ecliptic'},house:9,gate:31});
const spec=()=>({levels:levels(),human:human(),provenance:'user:explicit-test-bits'});

test('3/2/1 contains 3+2+1 INDEPENDENT one-bit binary positions',()=>{
  const result=projectThreeTwoOne(spec());
  assert.equal(result.schema,'synthia.321.binary.v2');
  assert.deepEqual(result.roleCounts,{level3:3,level2:2,level1:1});
  assert.deepEqual(result.bottomToTop,['level3','level2','level1']);
  assert.deepEqual(result.levels.level3.map(x=>x.bit),[1,0,1]);
  assert.ok(Object.values(result.levels).flat().every(x=>x.kind==='binary'&&[0,1].includes(x.bit)));
  assert.ok(!Object.hasOwn(result,'six'));
});

test('author-stated Human Design lowest groups are not silently converted to binary values',()=>{
  const groups=humanLowestLines(human());
  assert.equal(groups.length,3);
  assert.deepEqual(groups[0].components,{color:4,tone:2,base:5});
  assert.deepEqual(groups[1].components,{line:6});
  assert.deepEqual(groups[2].components,{arcAxis:{axis:'ecliptic'},house:9});
  assert.ok(groups.every(x=>!Object.hasOwn(x,'bit')));
  assert.equal(groups[2].status,'third-group-wording-needs-precision');
  assert.equal(projectThreeTwoOne(spec()).humanGroups[0].components.base,5);
});

test('rejects gate-only old projection and any inferred, missing, fractional or non-binary role',()=>{
  assert.throws(()=>projectThreeTwoOne({gate:3,provenance:'x'}),/explicit binary levels/);
  assert.throws(()=>projectThreeTwoOne({...spec(),provenance:null}),/provenance/);
  const bad=levels();bad.level3[0].bit=2;
  assert.throws(()=>projectThreeTwoOne({levels:bad,provenance:'test'}),/binary 0 or 1/);
  const missing=levels();missing.level2.pop();
  assert.throws(()=>projectThreeTwoOne({levels:missing,provenance:'test'}),/2 separate binary/);
  const fractional=levels();fractional.level1[0].bit=0.5;
  assert.throws(()=>projectThreeTwoOne({levels:fractional,provenance:'test'}),/binary 0 or 1/);
});

test('changing one binary does not recalculate other 3, 2, or 1 slots',()=>{
  const result=transitionThreeTwoOne(spec(),{level:'level3',index:2,toBit:1});
  assert.equal(result.changed,true);
  assert.deepEqual(result.before.levels.level3.map(x=>x.bit),[1,0,1]);
  assert.deepEqual(result.after.levels.level3.map(x=>x.bit),[1,1,1]);
  assert.deepEqual(result.after.levels.level2,result.before.levels.level2);
  assert.deepEqual(result.after.levels.level1,result.before.levels.level1);
  assert.throws(()=>transitionThreeTwoOne(spec(),{level:'level2',index:3,toBit:0}),/index/);
});

test('3/2/1 residents mount in original ConsciousnessSwarm without fabricated semantic links',()=>{
  const u=createUnifiedSwarm();const result=u.mount321({...spec(),prefix:'human'});
  assert.equal(result.nodeIds.length,6);
  assert.equal(result.semanticLinksAdded,0);
  assert.equal(u.swarm.positions().length,6);
  assert.equal(u.swarm.edges.size,0);
  const restored=UnifiedSwarm.restore(u.snapshot());
  assert.equal(restored.swarm.positions().length,6);
  assert.throws(()=>u.mount321({...spec(),prefix:'human'}),/already mounted/);
});

test('Zhouyi still maps 64 hexagrams source-faithfully but does NOT own the 3/2/1 structure',()=>{
  const seen=new Set();
  for(let gate=1;gate<=64;gate++){
    const out=projectZhouyiSixLine({gate,provenance:'source:gate-table'});
    assert.equal(out.six,fromLines(out.bits));
    assert.deepEqual(out.threeBigrams.flat(),out.bits);
    assert.deepEqual(out.twoTrigrams.flat(),out.bits);
    assert.equal(out.explicitlyNot,'synthia.321.binary.v2');
    seen.add(out.six);
  }
  assert.equal(seen.size,64);
});

test('Zhouyi 384 moving lines remain available as a DISTINCT operation',()=>{
  for(let gate=1;gate<=64;gate++)for(let line=1;line<=6;line++){
    const move=transitionZhouyiSixLine({gate,provenance:'source:gate-table'},{line});
    assert.equal(move.before.six^move.after.six,1<<(line-1));
  }
});
