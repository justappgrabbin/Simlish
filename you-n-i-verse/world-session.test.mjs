import test from 'node:test';
import assert from 'node:assert/strict';
import './vendor/kernel.js';
import './vendor/world-bridge.js';
import './vendor/mesh-assembly.js';
import './vendor/sim-core.js';
import { WorldSession } from './world-session.mjs';

test('Play changes actual Simlish object geometry through an executable ATO graph',async()=>{
  const session=new WorldSession();
  const original=session.snapshot();
  assert.equal(original.objects.length,6);
  const change=await session.perform({mode:'Play',targetId:'chair',op:'transform'});
  assert.equal(change.changed,true);
  assert.equal(change.event.type,'MORPH');
  assert.notEqual(change.prior.objects.find(x=>x.id==='chair').sy,
    change.after.objects.find(x=>x.id==='chair').sy);
  assert.equal(change.after.formationRevisions,2);
  assert.ok(change.formationArtifact);
  assert.ok(session.formation.history().some(e=>e.type==='formation-transformed'));
});

test('Create duplicates an actual object and composes the corresponding swarm particles',async()=>{
  const session=new WorldSession();
  const first=await session.perform({mode:'Create',targetId:'chair'});
  assert.equal(first.command.op,'duplicate');
  assert.equal(first.changed,true);
  assert.equal(first.after.objects.length,7);
  assert.equal(first.after.composites.length,1);
  assert.equal(first.after.composites[0].members[0],'chair');
  assert.ok(first.compositeId);
  const second=await session.perform({mode:'Create',targetId:'chair'});
  assert.equal(second.after.objects.length,8);
  assert.equal(second.after.composites.length,2);
  assert.notEqual(second.after.composites[1].members[1],first.after.composites[0].members[1]);
});

test('Mission runs the original particle physics and reports measured completion',async()=>{
  const session=new WorldSession();
  const fail=await session.perform({mode:'Mission',targetId:'plant'});
  assert.equal(fail.completed,false);
  assert.equal(fail.after.physicsTick,18);
  const pass=await session.perform({mode:'Mission',targetId:'chair'});
  assert.equal(pass.completed,true);
  assert.equal(pass.after.physicsTick,36);
  assert.ok(pass.event.fiveDimensionalState?.length===5);
});

test('Every distinct experience mode retains previous executable revisions',async()=>{
  const session=new WorldSession();
  await session.perform({mode:'Play',targetId:'chair'});
  const before=session.formation.describe().activeRevision;
  await session.perform({mode:'Create',targetId:'plant'});
  const after=session.formation.describe().activeRevision;
  assert.notEqual(after,before);
  const history=session.formation.history();
  assert.ok(history.some(e=>e.type==='formation-transformed'));
  assert.equal(session.commands.length,2);
});

test('Saved local commands replay exact world and physics state without a server',async()=>{
  const session=new WorldSession();
  await session.perform({mode:'Play',targetId:'table',op:'move'});
  await session.perform({mode:'Create',targetId:'chair'});
  await session.perform({mode:'Mission',targetId:'chair'});
  const saved=session.export();
  const restored=await WorldSession.fromCommands(saved);
  assert.deepEqual(restored.snapshot(),session.snapshot());
  assert.equal(restored.world.worldLedger.length,session.world.worldLedger.length);
  assert.equal(restored.physics.history.length,session.physics.history.length);
  assert.equal(restored.commands.length,3);
});

test('Invalid targets and operations cannot masquerade as successful world morphing',async()=>{
  const session=new WorldSession();
  await assert.rejects(()=>session.perform({mode:'Play',targetId:'missing',op:'transform'}),/unavailable/);
  await assert.rejects(()=>session.perform({mode:'Play',targetId:'chair',op:'universal'}),/Unknown physical operation/);
  assert.equal(session.commands.length,0);
  assert.equal(session.snapshot().formationRevisions,1);
});
