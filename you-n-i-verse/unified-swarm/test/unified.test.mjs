import test from 'node:test';
import assert from 'node:assert/strict';
import {createUnifiedSwarm,UnifiedSwarm} from '../src/unified-swarm.mjs';
import {frameFromAtlas} from '../src/spritesheet.mjs';
const canonical={planetary:1,dimension:'Movement',gate:3,line:1,color:1,tone:1,base:1,degree:1,minute:0,second:0,arcAxis:{arcUnit:1,axis:'Vertical'},zodiac:1,house:1};
test('four primitives and one external semantic authority',()=>{
  const source={structure:(s,c)=>({s,c})},u=createUnifiedSwarm({source});
  assert.deepEqual(u.primitives,['tool-factory','neural-net-maker','model-maker','foundry']);
  const result=u.inspect({text:'morph'},{view:'being'});assert.equal(result.c.addressMode,'structural');
  assert.throws(()=>createUnifiedSwarm().inspect('morph'),/CONNECTED_GENOTYPE_SOURCE_REQUIRED/);
});
test('meaningful relations require provenance and preserve node identity',()=>{
  const u=createUnifiedSwarm();u.add({id:'a',label:'movement',x:-.5,y:-.2});u.add({id:'b',label:'shape',x:.5,y:.3});
  assert.throws(()=>u.relate({from:'a',to:'b',predicate:'transforms'}),/provenance/);
  const e=u.relate({from:'a',to:'b',predicate:'transforms',provenance:'source-confirmed'});
  assert.equal(e.relation,'transforms');
  const c=u.cloud();assert.equal(c.nodes.length,2);assert.equal(c.edges.length,1);
  assert.ok(u.cloudSVG().includes('movement'));
  const formed=u.arrange({a:{x:.4,y:.4},b:{x:-.4,y:-.4}},{rate:.1,momentGain:0});
  assert.equal(formed.tick,1);assert.equal(u.swarm.positions().length,2);
});
test('cloud SVG escapes untrusted text',()=>{
  const u=createUnifiedSwarm();u.add({id:'x',label:'<script>alert(1)</script>'});
  const svg=u.cloudSVG();assert.ok(svg.includes('&lt;script&gt;'));assert.ok(!svg.includes('<script>'));
});
test('Pass05 existing recursive VM executes real arithmetic composition',async()=>{
  const u=createUnifiedSwarm();u.form({name:'transform',inputs:['value'],steps:[{id:'factor',op:'const',args:[],value:2},{id:'scaled',op:'mul',args:['value','factor']}],output:'scaled',ontology:{status:'supplied-program-only'}});
  const r=await u.executeFormation('transform',{value:9});assert.equal(r.value,18);assert.ok(r.trace.length>=2);
  const resumed=UnifiedSwarm.restore(u.snapshot());assert.equal((await resumed.executeFormation('transform',{value:7})).value,14);
});
test('Neural formation actually uses existing VM, not fake training',async()=>{
  const u=createUnifiedSwarm();const r=await u.neural({layers:[{weights:[[1,2],[3,4]],bias:[0,0],activation:'identity'}],input:[1,1]});
  assert.deepEqual(r.output,[4,6]);
});
test('Gate to hexagram uses real King Wen six-line map and does not assert semantic color laws',()=>{
  const u=createUnifiedSwarm();assert.throws(()=>u.expression({gate:1,value:3}),/provenance/);
  const r=u.expression({gate:1,provenance:'supplied-binary-map',value:3});assert.equal(r.source.address.hexagram6,63);
  assert.equal(r.schema,'synthia-trimodal-expression-v1');assert.ok(u.expressionSVG({gate:1,provenance:'supplied-binary-map',value:3}).includes('<svg'));
});
test('Atlas packs original RGBA pixels, transparency and timing',()=>{
  const u=createUnifiedSwarm();const frames=[{id:'a',width:2,height:1,data:Uint8Array.from([30,5,2,0,50,60,70,255]),durationMs:80},{id:'b',width:1,height:2,data:Uint8Array.from([1,2,3,4,5,6,7,8]),durationMs:120}];
  const atlas=u.sprites(frames,{columns:2,padding:2});
  assert.deepEqual([...frameFromAtlas(atlas,'a').data],[...frames[0].data]);
  assert.deepEqual([...frameFromAtlas(atlas,'b').data],[...frames[1].data]);
  assert.equal(frameFromAtlas(atlas,'b').durationMs,120);
  assert.throws(()=>u.sprites([{width:1,height:1,data:new Uint8Array(3)}]),/length/);
});
test('Owner-gated contextual dispatch never fabricates authority',async()=>{
  const u=createUnifiedSwarm();let calls=0;u.register({name:'tool.build',requires:['intent:build'],execute:async()=>(calls++,{artifact:'executed'})});
  assert.equal(u.choices({features:[]})[0].eligible,false);
  await assert.rejects(()=>u.dispatch('tool.build',{},{}),/CONTEXT_CONSTRAINT_UNSATISFIED/);
  const denied=await u.dispatch('tool.build',{}, {context:{features:['intent:build']}});
  assert.equal(denied.status,'blocked');assert.equal(calls,0);
  const yes=await u.dispatch('tool.build',{}, {context:{features:['intent:build']},ownerPolicy:async()=>true});
  assert.equal(yes.status,'executed');assert.equal(calls,1);
});
test('Live Simlish phenotype mutation is delegated to actual existing plug only',async()=>{
  let calls=0;const plug={morph:async args=>(calls++,{phenotype:{status:'executed-and-observed',expression:{objectId:args.objectId,operation:args.operation}},swarmReceipt:{ok:true},genotype:{address:args.genotypeAddress}})};
  const u=createUnifiedSwarm({simlishPlug:plug});
  await assert.rejects(()=>u.morphSimlish({intent:'morph',genotypeSource:'world-hash',genotypeAddress:canonical,objectId:'chair'}),/PROVENANCE/);
  const out=await u.morphSimlish({intent:'morph chair',genotypeSource:'owner-explicit',genotypeAddress:canonical,objectId:'chair',operation:'transform'});
  assert.equal(out.status,'executed-and-observed');assert.equal(calls,1);assert.equal(u.receipts.length,1);
});
test('Snapshots never silently discard formations or history',()=>{
  const u=createUnifiedSwarm();u.add({id:'a',label:'first'});const snap=u.snapshot();const v=UnifiedSwarm.restore(snap);
  assert.equal(v.swarm.nodes.size,1);assert.equal(v.snapshot().swarm.nodes[0].id,'a');
  assert.throws(()=>UnifiedSwarm.restore({schema:'invalid'}),/INVALID_SWARM_CHECKPOINT/);
});
